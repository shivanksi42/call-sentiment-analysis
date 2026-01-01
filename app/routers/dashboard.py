from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

from ..models.database import get_db, CallAnalysis
from ..models.schemas import DashboardMetrics

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/all-data")
async def get_all_dashboard_data(db: Session = Depends(get_db)):
    """
    OPTIMIZED: Single API call that returns ALL dashboard data.
    Reduces multiple API calls to just one for faster loading.
    """
    # Single query - load all records once
    records = db.query(CallAnalysis).order_by(CallAnalysis.call_date.desc()).all()
    
    if not records:
        return {
            "metrics": {
                "total_calls": 0,
                "avg_score": 0,
                "escalation_rate": 0,
                "avg_call_duration": 0,
                "positive_rate": 0
            },
            "sentiment_pie": {"labels": [], "values": []},
            "urgency_distribution": {"labels": [], "values": []},
            "agent_performance": [],
            "daily_trends": {"dates": [], "calls": [], "avg_scores": []},
            "category_scores": [],
            "escalation_risk": {"labels": [], "values": []},
            "recent_calls": []
        }
    
    # Calculate all metrics in one pass
    total_calls = len(records)
    total_score = 0
    total_duration = 0
    
    sentiment_counts = defaultdict(int)
    urgency_counts = defaultdict(int)
    escalation_risks = []
    agent_data = defaultdict(lambda: {"scores": [], "sentiments": defaultdict(int)})
    category_data = defaultdict(lambda: {"total_score": 0, "max_score": 0})
    risk_buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
    
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_data = defaultdict(lambda: {"calls": 0, "total_score": 0, "positive": 0, "negative": 0})
    
    for r in records:
        total_score += r.overall_percentage or 0
        total_duration += r.duration_seconds or 0
        
        # Agent data
        agent = r.agent_name or "Unknown"
        agent_data[agent]["scores"].append(r.overall_percentage or 0)
        
        # Sentiment data
        if r.customer_sentiment:
            sentiment = r.customer_sentiment.get("overall_sentiment", "Unknown")
            sentiment_counts[sentiment] += 1
            agent_data[agent]["sentiments"][sentiment] += 1
            
            urgency = r.customer_sentiment.get("urgency_level", "Unknown")
            urgency_counts[urgency] += 1
            
            risk = r.customer_sentiment.get("escalation_risk", 0)
            escalation_risks.append(risk)
            
            # Risk buckets
            if risk <= 20:
                risk_buckets["0-20%"] += 1
            elif risk <= 40:
                risk_buckets["20-40%"] += 1
            elif risk <= 60:
                risk_buckets["40-60%"] += 1
            elif risk <= 80:
                risk_buckets["60-80%"] += 1
            else:
                risk_buckets["80-100%"] += 1
        
        # Category scores
        if r.question_scores:
            for q in r.question_scores:
                cat = q.get("category", "Unknown")
                category_data[cat]["total_score"] += q.get("score", 0)
                category_data[cat]["max_score"] += q.get("max_score", 0)
        
        # Daily trends
        if r.call_date and r.call_date >= thirty_days_ago:
            day = r.call_date.strftime("%Y-%m-%d")
            daily_data[day]["calls"] += 1
            daily_data[day]["total_score"] += r.overall_percentage or 0
            if r.customer_sentiment:
                s = r.customer_sentiment.get("overall_sentiment", "")
                if s == "Positive":
                    daily_data[day]["positive"] += 1
                elif s == "Negative":
                    daily_data[day]["negative"] += 1
    
    # Calculate final metrics
    avg_score = total_score / total_calls if total_calls > 0 else 0
    avg_duration = total_duration / total_calls if total_calls > 0 else 0
    high_escalation = sum(1 for risk in escalation_risks if risk > 50)
    escalation_rate = (high_escalation / total_calls * 100) if total_calls > 0 else 0
    positive_rate = (sentiment_counts.get("Positive", 0) / total_calls * 100) if total_calls > 0 else 0
    
    # Format agent performance
    agent_performance = []
    for agent, data in agent_data.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        agent_performance.append({
            "agent": agent,
            "avg_score": round(avg, 2),
            "total_calls": len(data["scores"]),
            "positive_calls": data["sentiments"].get("Positive", 0),
            "negative_calls": data["sentiments"].get("Negative", 0)
        })
    agent_performance.sort(key=lambda x: -x["avg_score"])
    
    # Format category scores
    category_scores = []
    for cat, data in category_data.items():
        pct = (data["total_score"] / data["max_score"] * 100) if data["max_score"] > 0 else 0
        category_scores.append({
            "category": cat,
            "avg_percentage": round(pct, 2)
        })
    category_scores.sort(key=lambda x: -x["avg_percentage"])
    
    # Format daily trends
    dates = sorted(daily_data.keys())
    daily_trends = {
        "dates": dates,
        "calls": [daily_data[d]["calls"] for d in dates],
        "avg_scores": [daily_data[d]["total_score"] / daily_data[d]["calls"] if daily_data[d]["calls"] > 0 else 0 for d in dates],
        "positive": [daily_data[d]["positive"] for d in dates],
        "negative": [daily_data[d]["negative"] for d in dates]
    }
    
    # Recent calls for quick view
    recent_calls = [{
        "id": r.id,
        "agent_name": r.agent_name or "Unknown",
        "customer_name": r.customer_name or "Unknown",
        "call_date": r.call_date.isoformat() if r.call_date else None,
        "duration": r.duration_seconds,
        "overall_score": r.overall_percentage,
        "sentiment": r.customer_sentiment.get("overall_sentiment", "Unknown") if r.customer_sentiment else "Unknown",
        "resolution_status": r.resolution_status
    } for r in records[:10]]
    
    return {
        "metrics": {
            "total_calls": total_calls,
            "avg_score": round(avg_score, 2),
            "escalation_rate": round(escalation_rate, 2),
            "avg_call_duration": round(avg_duration, 2),
            "positive_rate": round(positive_rate, 2)
        },
        "sentiment_pie": {
            "labels": list(sentiment_counts.keys()),
            "values": list(sentiment_counts.values())
        },
        "urgency_distribution": {
            "labels": list(urgency_counts.keys()),
            "values": list(urgency_counts.values())
        },
        "agent_performance": agent_performance[:10],
        "daily_trends": daily_trends,
        "category_scores": category_scores,
        "escalation_risk": {
            "labels": list(risk_buckets.keys()),
            "values": list(risk_buckets.values())
        },
        "recent_calls": recent_calls
    }


# Keep individual endpoints for backwards compatibility but they're now optional
@router.get("/metrics")
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    """Get aggregated metrics (use /all-data instead for better performance)"""
    data = await get_all_dashboard_data(db)
    return DashboardMetrics(
        total_calls=data["metrics"]["total_calls"],
        avg_score=data["metrics"]["avg_score"],
        sentiment_distribution=dict(zip(data["sentiment_pie"]["labels"], data["sentiment_pie"]["values"])),
        urgency_distribution=dict(zip(data["urgency_distribution"]["labels"], data["urgency_distribution"]["values"])),
        escalation_rate=data["metrics"]["escalation_rate"],
        avg_call_duration=data["metrics"]["avg_call_duration"],
        top_issues=[],
        agent_performance=data["agent_performance"],
        daily_trends=[]
    )
