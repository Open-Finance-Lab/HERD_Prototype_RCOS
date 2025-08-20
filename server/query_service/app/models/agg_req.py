from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

class ExpertInput(BaseModel):
    expert_id: Optional[str] = None      
    answer: str = Field(..., min_length=1)
    router_confidence: float = 0.5
    scope: Optional[str] = ""
    evidence: Optional[List[str]] = None

class AggregatorConfigModel(BaseModel):
    alpha_softmax: float = 3.0
    sim_thresh: float = 0.82
    selection_budget: int = 24
    lambda_redundancy: float = 0.6
    use_nli: bool = False
    nli_topK: int = 40
    sections: Optional[List[str]] = None
    domain_defaults: Optional[List[str]] = None

class AggregateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    experts: Union[Dict[str, ExpertInput], List[ExpertInput]]
    config: Optional[AggregatorConfigModel] = None