import json
import os
from src.text_metrics import contain_score, exact_match_score, token_f1_score
from src.task_eval import compute_task_metrics, extract_gold_answer_text
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, llm_model, predictions_path):
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        # 修复：增加 encoding="utf-8" 解决 UnicodeDecodeError
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            prediction_results = json.load(f)
        return prediction_results
    
    def calculate_llm_accuracy(self,pre_answer,gold_ans):
        system_prompt = """You are an expert evaluator. 
        """
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        response = self.llm_model.infer([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        return contain_score(pre_answers, gold_ans)

    def calculate_exact_match(self, pre_answer, gold_ans):
        return exact_match_score(pre_answer, gold_ans)

    def calculate_token_f1(self, pre_answer, gold_ans):
        return token_f1_score(pre_answer, gold_ans)
            
    def evaluate_sig_sample(self,idx,prediction):
        pre_answer = prediction["pred_answer"]
        gold_ans = extract_gold_answer_text(prediction["gold_answer"], prediction.get("question_type", ""))
        question_type = prediction.get("question_type", "")
        # llm_acc = 0.0
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        exact_match = self.calculate_exact_match(pre_answer, gold_ans)
        summary_token_f1 = self.calculate_token_f1(pre_answer, gold_ans) if question_type == "summary" else None
        return idx, llm_acc, contain_acc, exact_match, summary_token_f1

    def evaluate(self,max_workers):
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        exact_match_scores = [0.0] * len(self.prediction_results)
        summary_token_f1_scores = [None] * len(self.prediction_results)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc, exact_match, summary_token_f1  = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                exact_match_scores[idx] = exact_match
                summary_token_f1_scores[idx] = summary_token_f1
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                self.prediction_results[idx]["exact_match"] = exact_match
                self.prediction_results[idx]["summary_token_f1"] = summary_token_f1
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)
        exact_match_accuracy = sum(exact_match_scores) / len(exact_match_scores)
        non_summary_exact_scores = [
            exact_match_scores[idx]
            for idx, pred in enumerate(self.prediction_results)
            if pred.get("question_type") != "summary"
        ]
        summary_token_f1_values = [x for x in summary_token_f1_scores if x is not None]
        summary_metrics = {
            "count": len(summary_token_f1_values),
            "token_f1": (sum(summary_token_f1_values) / len(summary_token_f1_values)) if summary_token_f1_values else 0.0,
        }
        task_metrics = compute_task_metrics(self.prediction_results)

        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        logger.info(f"  Exact Match: {exact_match_accuracy:.4f} ({sum(exact_match_scores)}/{len(exact_match_scores)})")
        if non_summary_exact_scores:
            logger.info(f"  Exact Match (Non-summary): {sum(non_summary_exact_scores) / len(non_summary_exact_scores):.4f}")
        logger.info(f"  Summary Token F1: {summary_metrics['token_f1']:.4f} ({summary_metrics['count']} samples)")
        logger.info(f"  Title EM: {task_metrics['title_em']:.4f}")
        logger.info(f"  First Figure Token EM: {task_metrics['first_figure_token_em']:.4f}")
        logger.info(f"  Summary Token F1 (Task): {task_metrics['summary_token_f1']:.4f}")
        logger.info(f"  Quant Token Match: {task_metrics['quant_token_match']:.4f}")
        logger.info(f"  Quant Token F1: {task_metrics['quant_token_f1']:.4f}")
        logger.info(f"  Quant Canonical Answer EM: {task_metrics['quant_canonical_answer_em']:.4f}")
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "llm_accuracy": llm_accuracy,
                "contain_accuracy": contain_accuracy,
                "exact_match_accuracy": exact_match_accuracy,
                "non_summary_exact_match_accuracy": (sum(non_summary_exact_scores) / len(non_summary_exact_scores)) if non_summary_exact_scores else 0.0,
                "summary_metrics": summary_metrics,
                "task_metrics": task_metrics,
            }, f, ensure_ascii=False, indent=4)
        return llm_accuracy, contain_accuracy
