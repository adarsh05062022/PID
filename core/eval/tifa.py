"""
TIFA (Hu et al., 2023, https://github.com/Yushi-Hu/tifa) -- text-to-image
faithfulness via VQA. The pieces of the `tifascore` package this repo needs,
ported here because `tifascore.vqa_models` imports every VQA backend at module
load (modelscope, LAVIS, promptcap, fairseq), which makes the package
un-importable unless all of them are installed. Logic is copied from the
package; only imports were made lazy and the pipeline was split into a
cacheable question stage and a scoring stage.

Pipeline, as in the paper / README:
  1. question generation -- the LLaMA-2 generator the TIFA authors released
     (tifa-benchmark/llama2_tifa_question_generation), the no-OpenAI-key
     alternative to GPT-3.5 in the README;
  2. question filtering -- UnifiedQA-v2 (allenai/unifiedqa-v2-t5-large-1363200)
     must reproduce the answer from the caption alone;
  3. VQA on the image -- mPLUG-large (the README's recommended model, via
     modelscope) or BLIP-large (HF, fallback), free-form answer mapped to the
     closest choice with SBERT;
  4. TIFA = mean over questions of [VQA choice == gold answer], per image,
     then averaged over images.
"""
import torch
from statistics import mean

CATEGORIES = ['object', 'human', 'animal', 'food', 'activity', 'attribute',
              'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']
QG_MODEL = 'tifa-benchmark/llama2_tifa_question_generation'
UNIFIEDQA_MODEL = 'allenai/unifiedqa-v2-t5-large-1363200'
SBERT_MODEL = 'sentence-transformers/all-mpnet-base-v2'
FREE_FORM_THRESHOLD = 0.6

VQA_MODELS = {
    'mplug-large': ('MPLUG', 'damo/mplug_visual-question-answering_coco_large_en'),
    'blip-large': ('BLIP', 'Salesforce/blip-vqa-capfilt-large'),
    'blip-base': ('BLIP', 'Salesforce/blip-vqa-base'),
}


# ----------------------------------------------------------------- 1. questions
def get_llama2_pipeline(model_name: str = QG_MODEL, device_map='auto'):
    import transformers
    return transformers.pipeline('text-generation', model=model_name,
                                 torch_dtype=torch.float16, device_map=device_map)


def create_qg_prompt(caption: str) -> str:
    intro = ('Given an image description, generate one or two multiple-choice questions '
             'that verifies if the image description is correct.\n'
             'Classify each concept into a type (object, human, animal, food, activity, '
             'attribute, counting, color, material, spatial, location, shape, other), '
             'and then generate a question for each type.\n')
    prompt = f'<s>[INST] <<SYS>>\n{intro}\n<</SYS>>\n\n'
    prompt += f'Description: {caption} [/INST] Entities:'
    return prompt


def llama2_completion(pipeline, caption: str) -> str:
    prompt = create_qg_prompt(caption)
    sequences = pipeline(prompt, do_sample=False, num_beams=5, num_return_sequences=1, max_length=512)
    output = sequences[0]['generated_text'][len(prompt):]
    return output.split('\n\n')[0]


def parse_resp(resp: str):
    resp = resp.split('\n')
    question_instances = []
    this_entity = this_type = this_question = this_choices = this_answer = None
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = this_choices = this_answer = None
    return question_instances


def get_llama2_question_and_answers(pipeline, caption: str) -> list[dict]:
    resp = llama2_completion(pipeline, caption)
    out = []
    for entity, question, choices, answer, etype in parse_resp(resp):
        if etype not in CATEGORIES:
            continue
        if etype in ('animal', 'human'):
            etype = 'animal/human'
        out.append({'caption': caption, 'element': entity, 'question': question,
                    'choices': choices, 'answer': answer, 'element_type': etype})
    return out


# ------------------------------------------------------------------- 2. filter
class UnifiedQAModel:
    def __init__(self, model_name: str = UNIFIEDQA_MODEL):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def run_model(self, input_string, **generator_args):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors='pt')
            res = self.model.generate(input_ids.to(self.model.device), max_new_tokens=30, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def qa(self, question, context):
        answer = self.run_model(f'{question} \n {context}')[0]
        return ''.join(c for c in answer if c.isalnum() or c.isspace()).strip().lower()

    def mcqa(self, question, context, choices=('yes', 'no')):
        choice_text = ''
        for heading, choice in zip(['(A)', '(B)', '(C)', '(D)'], choices):
            choice_text += f'{heading} {choice} '
        return self.run_model(f'{question} \n {context} \n {choice_text}')[0]


def compute_prf(gold, pred) -> float:
    if len(gold) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    tp = sum(1 for g in gold if g in pred)
    fn = len(gold) - tp
    fp = sum(1 for p in pred if p not in gold)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def filter_question_and_answers(qa_model: UnifiedQAModel, caption_qas: list[dict]) -> list[dict]:
    from word2number import w2n
    kept, seen = [], set()
    for qa in caption_qas:
        if qa['question'] in seen:
            continue
        seen.add(qa['question'])
        if qa_model.mcqa(qa['question'], qa['caption'], choices=qa['choices']) != qa['answer']:
            continue
        if qa['answer'] not in ('yes', 'no'):
            free_form = qa_model.qa(qa['question'], qa['caption']).strip()
            gold = qa['answer']
            if gold.isnumeric():
                try:
                    free_form = str(w2n.word_to_num(free_form))
                except Exception:
                    pass
            if compute_prf(gold.split(), free_form.split()) <= FREE_FORM_THRESHOLD:
                continue
        kept.append(qa)
    return kept


# ---------------------------------------------------------------------- 3. VQA
class SBERTModel:
    def __init__(self, ckpt: str = SBERT_MODEL):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).eval()
        if torch.cuda.is_available():
            self.model.cuda()

    @torch.no_grad()
    def embed_sentences(self, sentences):
        enc = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        tok = self.model(**enc)[0]
        mask = enc['attention_mask'].unsqueeze(-1).expand(tok.size()).float()
        emb = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(emb, p=2, dim=1).cpu()

    def multiple_choice(self, answer, choices):
        a = self.embed_sentences([answer])
        c = self.embed_sentences(list(choices))
        return choices[int(torch.argmax(c @ a.T))]


class MPLUG:
    def __init__(self, ckpt):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt)

    def vqa(self, image_path, question):
        return self.pipeline_vqa({'image': image_path, 'question': question})['text']


class BLIP:
    def __init__(self, ckpt):
        from transformers import AutoProcessor, BlipForQuestionAnswering
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    def vqa(self, image_path, question):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, text=question, return_tensors='pt').to(self.device)
        ids = self.model.generate(**inputs, max_length=50)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


class VQAModel:
    def __init__(self, model_name: str = 'mplug-large'):
        self.model_name = model_name
        class_name, ckpt = VQA_MODELS[model_name]
        self.model = {'MPLUG': MPLUG, 'BLIP': BLIP}[class_name](ckpt)
        self.sbert_model = SBERTModel()

    @torch.no_grad()
    def multiple_choice_vqa(self, image_path, question, choices):
        free_form = self.model.vqa(image_path, question)
        mc = free_form if free_form in choices else self.sbert_model.multiple_choice(free_form, choices)
        return {'free_form_answer': free_form, 'multiple_choice_answer': mc}


# -------------------------------------------------------------------- 4. score
def tifa_score_single(vqa_model: VQAModel, question_answer_pairs: list[dict], img_path: str) -> dict:
    scores, logs = [], {}
    for qa in question_answer_pairs:
        ans = vqa_model.multiple_choice_vqa(img_path, qa['question'], qa['choices'])
        score = int(ans['multiple_choice_answer'] == qa['answer'])
        logs[qa['question']] = dict(qa, free_form_vqa=ans['free_form_answer'],
                                    multiple_choice_vqa=ans['multiple_choice_answer'], scores=score)
        scores.append(score)
    return {'tifa_score': mean(scores) if scores else None, 'question_details': logs}
