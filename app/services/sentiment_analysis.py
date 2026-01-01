"""
Async Sentiment Analysis Service

Production-ready async sentiment analysis using OpenAI SDK v1.
Features: parallel execution, retries, concurrency control, timeouts.
"""
import os
import json
import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Callable, Coroutine

from openai import AsyncOpenAI  # OpenAI SDK v1 async client

# Import your existing schema classes (assumed present in your project)
from ..models.schemas import (
    CustomerSentiment,
    AgentBehavior,
    ComplianceRisk,
    QuestionScore,
    SentimentType,
    UrgencyLevel,
    EmotionType,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OpenAIAPIError(Exception):
    """Wrapper for OpenAI related errors when we want to signal failures explicitly."""


class AsyncSentimentAnalysisService:
    """
    Production-ready async sentiment analysis service that talks to OpenAI Async SDK v1.

    Features:
    - Lazy AsyncOpenAI client creation
    - Concurrency limiting via asyncio.Semaphore
    - Retries with exponential backoff for transient errors (rate-limits / timeouts)
    - Per-call timeouts
    - Optional streaming hook
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_concurrent_requests: int = 8,
        request_timeout: float = 20.0,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ):
        """
        Args:
            api_key: If None, will read from OPENAI_API_KEY env var lazily.
            model: OpenAI model to use.
            max_concurrent_requests: semaphore size to limit concurrency.
            request_timeout: timeout for each API call (seconds).
            max_retries: max retry attempts for transient errors.
            initial_backoff: base backoff in seconds.
            max_backoff: cap for backoff.
        """
        self._client: Optional[AsyncOpenAI] = None
        self._api_key = api_key  # Will be loaded lazily if None
        self.model = model

        # Controls
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.initial_backoff = float(initial_backoff)
        self.max_backoff = float(max_backoff)

        # common system role messages per analysis type (keeps prompts DRY)
        self._system_roles = {
            "sentiment": "You are a sentiment analysis expert for call center quality auditing. Always respond with valid JSON only.",
            "agent_behavior": "You are a call quality expert. Always respond with valid JSON only.",
            "compliance": "You are a compliance risk expert. Always respond with valid JSON only.",
            "questionnaire": "You are a call quality auditor. Score calls fairly based on the evidence in the transcription. Always respond with valid JSON only.",
            "summary": "You are a call summarization expert. Be concise and factual.",
            "intent": "You are a customer intent analysis expert. Always respond with valid JSON only.",
        }

    async def _ensure_client(self) -> AsyncOpenAI:
        """Lazy init the AsyncOpenAI client."""
        if self._client is None:
            # Load API key lazily
            api_key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self._client = AsyncOpenAI(api_key=api_key)
            logger.info("Initialized AsyncOpenAI client")
        return self._client

    async def _backoff_sleep(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        base = self.initial_backoff * (2 ** attempt)
        wait = min(self.max_backoff, base)
        # add small jitter
        jitter = wait * 0.1 * (2 * (math.random() if hasattr(math, "random") else 0) - 1)
        # To avoid dependency on random module for small jitter, use a deterministic small jitter fallback
        # but ideally use random.random() if available. We'll use a safe no-jitter fallback if not available.
        try:
            import random

            jitter = wait * 0.1 * (random.random() - 0.5)
        except Exception:
            jitter = 0.0
        sleep_for = max(0.0, wait + jitter)
        logger.debug("Backoff sleeping for %.2fs (attempt=%d)", sleep_for, attempt)
        await asyncio.sleep(sleep_for)

    async def _is_transient_error(self, exc: Exception) -> bool:
        """Heuristic to identify transient (retryable) errors."""
        # Common transient errors include network timeouts, rate limits, temporary connection issues.
        # We don't import SDK-specific error classes to keep compatibility; inspect messages instead.
        message = str(exc).lower()
        if isinstance(exc, asyncio.TimeoutError):
            return True
        if "rate limit" in message or "too many requests" in message or "429" in message:
            return True
        if "connection" in message or "connection reset" in message or "timed out" in message:
            return True
        # Add other heuristics as needed
        return False

    async def _call_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        timeout: Optional[float] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Centralized call wrapper to OpenAI chat completions with retries, timeout and concurrency limits.

        If stream=True, the OpenAI SDK may return a streaming response; our wrapper supports a simple callback.
        """
        client = await self._ensure_client()
        attempt = 0
        timeout = timeout or self.request_timeout

        while attempt <= self.max_retries:
            attempt += 1
            try:
                async with self._semaphore:
                    coro = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                    # enforce per-call timeout
                    response = await asyncio.wait_for(coro, timeout=timeout)

                    # If streaming with callback: SDK behavior varies; here we assume SDK yields a stream object
                    # that is already handled by the SDK. If you need to handle streamed chunks manually, adapt here.
                    return response
            except Exception as exc:
                # classify transient errors to retry
                transient = await self._is_transient_error(exc)
                logger.warning(
                    "OpenAI call failed on attempt %d/%d: %s (transient=%s)",
                    attempt,
                    self.max_retries,
                    exc,
                    transient,
                )
                if attempt > self.max_retries or not transient:
                    logger.exception("Max retries reached or not transient—raising OpenAIAPIError")
                    raise OpenAIAPIError(f"OpenAI call failed: {exc}") from exc
                # backoff with jitter
                backoff_seconds = min(self.max_backoff, self.initial_backoff * (2 ** (attempt - 1)))
                # add small jitter using random
                try:
                    import random

                    jitter = random.uniform(-0.1 * backoff_seconds, 0.1 * backoff_seconds)
                except Exception:
                    jitter = 0.0
                sleep_for = max(0.0, backoff_seconds + jitter)
                logger.info("Retrying after %.2fs", sleep_for)
                await asyncio.sleep(sleep_for)

    # ---------------------
    # Internal helpers
    # ---------------------
    def _safe_json_loads(self, text: str) -> Any:
        """Try to parse JSON robustly; if invalid, attempt to extract JSON substring, else raise."""
        if not text or text.strip() == "":
            raise ValueError("Empty response content")
        
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object {...}
            obj_start = text.find("{")
            obj_end = text.rfind("}")
            
            # Try to find JSON array [...]
            arr_start = text.find("[")
            arr_end = text.rfind("]")
            
            # Determine which comes first and is valid
            candidates = []
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                candidates.append((obj_start, obj_end, "{"))
            if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                candidates.append((arr_start, arr_end, "["))
            
            # Sort by start position and try each
            for start, end, _ in sorted(candidates, key=lambda x: x[0]):
                substring = text[start : end + 1]
                try:
                    return json.loads(substring)
                except json.JSONDecodeError:
                    continue
            
            logger.debug("Failed parsing JSON from response: %s", text[:200])
            raise

    # ---------------------
    # Public analysis methods
    # ---------------------
    async def analyze_call(self, transcription: str) -> Dict[str, Any]:
        """
        Main entrypoint — runs multiple analyses in parallel where safe and returns combined result.
        """
        if not transcription:
            raise ValueError("transcription must be provided")

        # Run analysis parts concurrently to save wall-clock time (limited by semaphore)
        # We'll run sentiment, agent behavior, compliance, intent and summary concurrently.
        tasks = [
            asyncio.create_task(self._analyze_sentiment(transcription)),
            asyncio.create_task(self._analyze_agent_behavior(transcription)),
            asyncio.create_task(self._assess_compliance_risk(transcription)),
            asyncio.create_task(self._analyze_intent(transcription)),
            asyncio.create_task(self._generate_summary(transcription)),
            asyncio.create_task(self._score_questionnaire(transcription)),
        ]

        # Gather all results and handle exceptions per-task
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results carefully and handle exceptions gracefully
        sentiment_result = results[0] if not isinstance(results[0], Exception) else None
        agent_behavior = results[1] if not isinstance(results[1], Exception) else None
        compliance_risk = results[2] if not isinstance(results[2], Exception) else None
        intent_analysis = results[3] if not isinstance(results[3], Exception) else {
            "intent": "Unknown",
            "issues": [],
            "resolution_status": "Unknown",
            "follow_up_required": False,
        }
        call_summary = results[4] if not isinstance(results[4], Exception) else "Summary unavailable"
        question_scores = results[5] if not isinstance(results[5], Exception) else []

        # Compose final payload (maintaining the same keys as your original service)
        return {
            "customer_sentiment": sentiment_result,
            "agent_behavior": agent_behavior,
            "compliance_risk": compliance_risk,
            "question_scores": question_scores,
            "call_summary": call_summary,
            "customer_intent": intent_analysis.get("intent"),
            "key_issues": intent_analysis.get("issues"),
            "resolution_status": intent_analysis.get("resolution_status"),
            "follow_up_required": intent_analysis.get("follow_up_required"),
        }

    # ---------------------
    # Individual analysis implementations
    # ---------------------
    def _get_questionnaire(self) -> List[Dict[str, Any]]:
        """Questionnaire kept as a pure function so it can be reused/tested easily."""
        return [
            # Call Opening
            {"category": "Call Opening", "question": "Did agent probe customer name before continuing?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent open call as per timelines and script?", "max_score": 3},
            {"category": "Call Opening", "question": "Did agent give opening within 5 seconds?", "max_score": 2},
            {"category": "Call Opening", "question": "Did agent greet according to language selection?", "max_score": 2},
            # Soft Skills
            {"category": "Soft Skills", "question": "Did agent willingly help without making commitments?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent use proper sentence structure and grammar?", "max_score": 3},
            {"category": "Soft Skills", "question": "Was agent confident during the call?", "max_score": 3},
            {"category": "Soft Skills", "question": "Did agent show empathy towards customer?", "max_score": 4},
            {"category": "Soft Skills", "question": "Did agent maintain professional tone throughout?", "max_score": 3},
            # Probing & Understanding
            {"category": "Probing & Understanding", "question": "Did agent ask effective questions to understand needs?", "max_score": 4},
            {"category": "Probing & Understanding", "question": "Did agent understand customer concern at first instance?", "max_score": 3},
            {"category": "Probing & Understanding", "question": "Did agent ask pertinent diagnostic questions?", "max_score": 3},
            # Problem Resolution
            {"category": "Problem Resolution", "question": "Did agent provide accurate information?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent offer appropriate solutions?", "max_score": 5},
            {"category": "Problem Resolution", "question": "Did agent handle objections effectively?", "max_score": 4},
            # Call Closing
            {"category": "Call Closing", "question": "Did agent follow correct closing format?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent summarize the call properly?", "max_score": 3},
            {"category": "Call Closing", "question": "Did agent ask for further assistance?", "max_score": 2},
            # Critical Parameters
            {"category": "Critical Parameters", "question": "Did agent NOT disconnect without warning?", "max_score": 10},
            {"category": "Critical Parameters", "question": "Did agent use correct categorization?", "max_score": 5},
        ]

    async def _analyze_sentiment(self, transcription: str) -> CustomerSentiment:
        prompt = f"""Analyze the customer sentiment from this call transcription and return a JSON object with these exact fields:
- overall_sentiment: one of "Positive", "Neutral", "Negative", "Mixed"
- emotions: array of emotions detected, each one of: "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"
- urgency_level: one of "High", "Medium", "Low"
- frustration_indicator: boolean
- escalation_risk: number between 0-100 representing percentage
- call_opening_emotion: one of "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"
- call_end_emotion: one of "Calm", "Cooperative", "Confused", "Angry", "Frustrated", "Satisfied"

Transcription: {transcription}

Return ONLY valid JSON, no other text.
"""
        messages = [
            {"role": "system", "content": self._system_roles["sentiment"]},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = await self._call_chat_completion(messages, timeout=self.request_timeout)
            # Access returned text (SDK may vary in attribute names; adjust if needed)
            content = getattr(resp.choices[0].message, "content", None) or getattr(resp.choices[0].message, "content", None) or str(resp)
            # If content is an object with 'content' under choices[0].message['content'] depending on SDK representation:
            if isinstance(content, dict) and "content" in content:
                content = content["content"]
            parsed = self._safe_json_loads(content)
            return CustomerSentiment(
                overall_sentiment=SentimentType(parsed.get("overall_sentiment", "Neutral")),
                emotions=[EmotionType(e) for e in parsed.get("emotions", ["Calm"])],
                urgency_level=UrgencyLevel(parsed.get("urgency_level", "Medium")),
                frustration_indicator=bool(parsed.get("frustration_indicator", False)),
                escalation_risk=int(parsed.get("escalation_risk", 0)),
                call_opening_emotion=EmotionType(parsed.get("call_opening_emotion", "Calm")),
                call_end_emotion=EmotionType(parsed.get("call_end_emotion", "Calm")),
            )
        except Exception as exc:
            logger.exception("Sentiment analysis failed, returning defaults: %s", exc)
            return CustomerSentiment(
                overall_sentiment=SentimentType.NEUTRAL,
                emotions=[EmotionType.CALM],
                urgency_level=UrgencyLevel.MEDIUM,
                frustration_indicator=False,
                escalation_risk=0,
                call_opening_emotion=EmotionType.CALM,
                call_end_emotion=EmotionType.CALM,
            )

    async def _analyze_agent_behavior(self, transcription: str) -> AgentBehavior:
        prompt = f"""Analyze the agent's behavior in this call transcription and return a JSON object with these exact boolean fields:
- calmness: was the agent calm throughout?
- confidence: did the agent sound confident?
- politeness: was the agent polite?
- empathy: did the agent show empathy?
- proper_grammar: did the agent use proper grammar?

Transcription: {transcription}

Return ONLY valid JSON, no other text.
"""
        messages = [
            {"role": "system", "content": self._system_roles["agent_behavior"]},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._call_chat_completion(messages, timeout=self.request_timeout)
            content = getattr(resp.choices[0].message, "content", None) or str(resp)
            parsed = self._safe_json_loads(content)
            return AgentBehavior(
                calmness=bool(parsed.get("calmness", True)),
                confidence=bool(parsed.get("confidence", True)),
                politeness=bool(parsed.get("politeness", True)),
                empathy=bool(parsed.get("empathy", True)),
                proper_grammar=bool(parsed.get("proper_grammar", True)),
            )
        except Exception:
            logger.exception("Agent behavior analysis failed")
            return AgentBehavior(
                calmness=True, confidence=True, politeness=True, empathy=True, proper_grammar=True
            )

    async def _assess_compliance_risk(self, transcription: str) -> ComplianceRisk:
        prompt = f"""Assess the compliance risk in this call transcription and return a JSON object with:
- fraud_suspected: boolean indicating if fraud is suspected
- compliance_risk: one of "low", "medium", "high"
- trust_justification: brief explanation of the risk assessment

Transcription: {transcription}

Return ONLY valid JSON, no other text.
"""
        messages = [
            {"role": "system", "content": self._system_roles["compliance"]},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._call_chat_completion(messages, timeout=self.request_timeout)
            content = getattr(resp.choices[0].message, "content", None) or str(resp)
            parsed = self._safe_json_loads(content)
            return ComplianceRisk(
                fraud_suspected=bool(parsed.get("fraud_suspected", False)),
                compliance_risk=str(parsed.get("compliance_risk", "low")),
                trust_justification=str(parsed.get("trust_justification", "No concerns identified")),
            )
        except Exception:
            logger.exception("Compliance risk assessment failed")
            return ComplianceRisk(fraud_suspected=False, compliance_risk="low", trust_justification="Unable to assess")

    async def _score_questionnaire(self, transcription: str) -> List[QuestionScore]:
        questionnaire = self._get_questionnaire()
        questions_text = "\n".join([f"{i+1}. [{q['category']}] {q['question']} (max: {q['max_score']} points)" for i, q in enumerate(questionnaire)])
        prompt = f"""Score this call against each question. For each question, provide:
- score: points earned (0 to max_score)
- answer: brief explanation (Yes/No/NA with reason)

Questions:
{questions_text}

Transcription: {transcription}

Return a JSON array with objects containing: category, question, answer, score, max_score
Return ONLY valid JSON array, no other text.
"""
        messages = [
            {"role": "system", "content": self._system_roles["questionnaire"]},
            {"role": "user", "content": prompt},
        ]
        try:
            # Questionnaire needs more tokens (20 questions with detailed responses)
            resp = await self._call_chat_completion(messages, timeout=30.0, max_tokens=2500)
            content = getattr(resp.choices[0].message, "content", None) or ""
            
            # Log if content seems problematic
            if not content or len(content) < 10:
                logger.warning("Questionnaire got short/empty response: %s", repr(content))
            results = self._safe_json_loads(content)
            output = []
            for r in results:
                try:
                    score = int(r.get("score", 0))
                    max_score = int(r.get("max_score", 5))
                except Exception:
                    score = 0
                    max_score = r.get("max_score", 5) if isinstance(r.get("max_score", None), int) else 5
                output.append(
                    QuestionScore(
                        category=r.get("category", "Unknown"),
                        question=r.get("question", ""),
                        answer=r.get("answer", "NA"),
                        score=min(score, max_score),
                        max_score=max_score,
                    )
                )
            return output
        except Exception:
            logger.exception("Questionnaire scoring failed - returning defaults")
            return [
                QuestionScore(
                    category=q["category"],
                    question=q["question"],
                    answer="Unable to assess",
                    score=0,
                    max_score=q["max_score"],
                )
                for q in questionnaire
            ]

    async def _generate_summary(self, transcription: str) -> str:
        prompt = f"""Provide a brief 2-3 sentence summary of this customer service call:

{transcription}

Focus on: the customer's issue, what action was taken, and the outcome.
"""
        messages = [
            {"role": "system", "content": self._system_roles["summary"]},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._call_chat_completion(messages, timeout=self.request_timeout)
            # SDK response object shape may vary; support common access patterns
            content = getattr(resp.choices[0].message, "content", None) or str(resp)
            # If content is dict with 'content' key:
            if isinstance(content, dict) and "content" in content:
                return content["content"].strip()
            return str(content).strip()
        except Exception:
            logger.exception("Summary generation failed")
            return "Summary not available."

    async def _analyze_intent(self, transcription: str) -> Dict[str, Any]:
        prompt = f"""Analyze this call and return a JSON object with:
- intent: customer's primary intent (e.g., "Complaint", "Query", "Feedback", "Request")
- issues: array of specific issues raised
- resolution_status: one of "Resolved", "Partially Resolved", "Unresolved", "Requires Follow-up"
- follow_up_required: boolean

Transcription: {transcription}

Return ONLY valid JSON, no other text.
"""
        messages = [
            {"role": "system", "content": self._system_roles["intent"]},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._call_chat_completion(messages, timeout=self.request_timeout)
            content = getattr(resp.choices[0].message, "content", None) or str(resp)
            parsed = self._safe_json_loads(content)
            return {
                "intent": parsed.get("intent", "Unknown"),
                "issues": parsed.get("issues", []),
                "resolution_status": parsed.get("resolution_status", "Unknown"),
                "follow_up_required": bool(parsed.get("follow_up_required", False)),
            }
        except Exception:
            logger.exception("Intent analysis failed")
            return {"intent": "Unknown", "issues": [], "resolution_status": "Unknown", "follow_up_required": False}


# Singleton instance for use throughout the app
sentiment_service = AsyncSentimentAnalysisService()
