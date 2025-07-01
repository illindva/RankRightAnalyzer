from typing import Dict, List, Any
from azure_openai_client import AzureOpenAIClient

class EvaluationEngine:
    """Engine for evaluating documents against predefined criteria"""
    
    def __init__(self, ai_client: AzureOpenAIClient):
        self.ai_client = ai_client
        self.criteria = self._initialize_criteria()
    
    def _initialize_criteria(self) -> Dict[str, str]:
        """Initialize the 6 predefined evaluation criteria"""
        
        return {
            "Clarity & Readability": """
            Evaluate how clear, understandable, and accessible the content is to its intended audience. 
            Consider language complexity, sentence structure, jargon usage, and overall readability. 
            Assess whether the content effectively communicates its message without ambiguity.
            """,
            
            "Completeness & Coverage": """
            Assess whether the document thoroughly covers all necessary topics and provides comprehensive information. 
            Evaluate if all required sections are present, if key information is missing, and whether the depth 
            of coverage is appropriate for the document's purpose.
            """,
            
            "Accuracy & Reliability": """
            Evaluate the factual accuracy, consistency, and reliability of the information presented. 
            Look for contradictions, outdated information, unsupported claims, and verify that 
            statements are backed by appropriate evidence or sources.
            """,
            
            "Structure & Organization": """
            Assess the logical flow, organization, and structure of the document. 
            Evaluate heading hierarchy, paragraph organization, use of lists and tables, 
            and whether the content follows a logical sequence that aids comprehension.
            """,
            
            "Compliance & Standards": """
            Evaluate adherence to relevant standards, regulations, policies, or industry best practices. 
            Assess whether the document meets organizational requirements, follows established guidelines, 
            and complies with applicable regulatory or quality standards.
            """,
            
            "Actionability & Usefulness": """
            Assess how practical and useful the document is for its intended purpose. 
            Evaluate whether it provides clear guidance, actionable steps, practical examples, 
            and whether readers can effectively use the information to achieve desired outcomes.
            """
        }
    
    def evaluate_content(self, content: str) -> Dict[str, Any]:
        """
        Evaluate content against all predefined criteria.
        
        Args:
            content: Text content to evaluate
            
        Returns:
            Dictionary containing evaluation results for each criterion
        """
        
        results = {}
        
        for criterion_name, criterion_description in self.criteria.items():
            try:
                result = self.ai_client.evaluate_against_criteria(
                    content=content,
                    criterion_name=criterion_name,
                    criterion_description=criterion_description
                )
                results[criterion_name] = result
                
            except Exception as e:
                # Provide fallback result if evaluation fails
                results[criterion_name] = {
                    "ranking": "Amber",
                    "score": 5.0,
                    "explanation": f"Evaluation failed: {str(e)}",
                    "key_findings": ["Unable to complete automated evaluation"],
                    "recommendations": ["Manual review required due to evaluation error"]
                }
        
        return results
    
    def evaluate_single_criterion(self, content: str, criterion_name: str) -> Dict[str, Any]:
        """
        Evaluate content against a single criterion.
        
        Args:
            content: Text content to evaluate
            criterion_name: Name of the criterion to evaluate against
            
        Returns:
            Dictionary containing evaluation result
        """
        
        if criterion_name not in self.criteria:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        
        criterion_description = self.criteria[criterion_name]
        
        return self.ai_client.evaluate_against_criteria(
            content=content,
            criterion_name=criterion_name,
            criterion_description=criterion_description
        )
    
    def get_criteria_names(self) -> List[str]:
        """Get list of all criterion names"""
        return list(self.criteria.keys())
    
    def get_criteria_descriptions(self) -> Dict[str, str]:
        """Get all criteria with their descriptions"""
        return self.criteria.copy()
    
    def get_criterion_description(self, criterion_name: str) -> str:
        """Get description for a specific criterion"""
        return self.criteria.get(criterion_name, "")
    
    def calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall document score based on all criteria.
        
        Args:
            evaluation_results: Results from evaluate_content()
            
        Returns:
            Dictionary containing overall assessment
        """
        
        if not evaluation_results:
            return {
                "overall_score": 0.0,
                "overall_ranking": "Red",
                "total_criteria": 0,
                "green_count": 0,
                "amber_count": 0,
                "red_count": 0
            }
        
        scores = []
        ranking_counts = {"Green": 0, "Amber": 0, "Red": 0}
        
        for criterion_result in evaluation_results.values():
            scores.append(criterion_result["score"])
            ranking = criterion_result["ranking"]
            if ranking in ranking_counts:
                ranking_counts[ranking] += 1
        
        # Calculate overall score as average
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine overall ranking based on average score
        if overall_score >= 8.0:
            overall_ranking = "Green"
        elif overall_score >= 5.0:
            overall_ranking = "Amber"
        else:
            overall_ranking = "Red"
        
        return {
            "overall_score": round(overall_score, 2),
            "overall_ranking": overall_ranking,
            "total_criteria": len(evaluation_results),
            "green_count": ranking_counts["Green"],
            "amber_count": ranking_counts["Amber"],
            "red_count": ranking_counts["Red"],
            "individual_scores": scores
        }
    
    def generate_improvement_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """
        Generate prioritized improvement recommendations based on evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_content()
            
        Returns:
            List of prioritized recommendations
        """
        
        recommendations = []
        
        # Collect all recommendations and prioritize by ranking
        red_recommendations = []
        amber_recommendations = []
        
        for criterion_name, result in evaluation_results.items():
            ranking = result["ranking"]
            criterion_recommendations = result.get("recommendations", [])
            
            for rec in criterion_recommendations:
                formatted_rec = f"[{criterion_name}] {rec}"
                
                if ranking == "Red":
                    red_recommendations.append(formatted_rec)
                elif ranking == "Amber":
                    amber_recommendations.append(formatted_rec)
        
        # Prioritize Red issues first, then Amber
        recommendations.extend(red_recommendations[:3])  # Top 3 critical issues
        recommendations.extend(amber_recommendations[:2])  # Top 2 amber issues
        
        return recommendations
