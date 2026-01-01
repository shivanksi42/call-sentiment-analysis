from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SentimentType(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    MIXED = "Mixed"


class UrgencyLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EmotionType(str, Enum):
    CALM = "Calm"
    COOPERATIVE = "Cooperative"
    CONFUSED = "Confused"
    ANGRY = "Angry"
    FRUSTRATED = "Frustrated"
    SATISFIED = "Satisfied"


class CallType(str, Enum):
    INCOMING = "Incoming"
    OUTGOING = "Outgoing"


# Request Models
class CallUploadRequest(BaseModel):
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    call_type: CallType = CallType.INCOMING


# Response Models
class TranscriptionResult(BaseModel):
    text: str
    duration: float
    language: str


class QuestionScore(BaseModel):
    category: str
    question: str
    answer: str
    score: int
    max_score: int


class AgentBehavior(BaseModel):
    calmness: bool
    confidence: bool
    politeness: bool
    empathy: bool
    proper_grammar: bool


class CustomerSentiment(BaseModel):
    overall_sentiment: SentimentType
    emotions: List[EmotionType]
    urgency_level: UrgencyLevel
    frustration_indicator: bool
    escalation_risk: float  # 0-100%
    call_opening_emotion: EmotionType
    call_end_emotion: EmotionType


class ComplianceRisk(BaseModel):
    fraud_suspected: bool
    compliance_risk: str  # low, medium, high
    trust_justification: str


class CallAnalysisResult(BaseModel):
    call_id: str
    call_date: datetime
    audit_date: datetime
    duration_seconds: float
    
    # Agent Info
    agent_id: Optional[str]
    agent_name: Optional[str]
    
    # Customer Info
    customer_name: Optional[str]
    customer_phone: Optional[str]
    
    # Transcription
    transcription: str
    language: str
    
    # Analysis Results
    call_summary: str
    customer_sentiment: CustomerSentiment
    agent_behavior: AgentBehavior
    compliance_risk: ComplianceRisk
    
    # Scoring
    question_scores: List[QuestionScore]
    total_score: int
    max_score: int
    overall_percentage: float
    
    # Intent & Insights
    customer_intent: str
    key_issues: List[str]
    resolution_status: str
    follow_up_required: bool


class DashboardMetrics(BaseModel):
    total_calls: int
    avg_score: float
    sentiment_distribution: Dict[str, int]
    urgency_distribution: Dict[str, int]
    escalation_rate: float
    avg_call_duration: float
    top_issues: List[Dict[str, Any]]
    agent_performance: List[Dict[str, Any]]
    daily_trends: List[Dict[str, Any]]


class CallRecord(BaseModel):
    id: str
    agent_name: str
    customer_name: str
    call_date: datetime
    duration: float
    overall_score: float
    sentiment: str
    status: str

