import os
import uuid
import tempfile
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Header
from sqlalchemy.orm import Session
from typing import Optional, List

from ..models.database import get_db, CallAnalysis
from ..models.schemas import CallAnalysisResult, CallType
from ..services.transcription import transcription_service
from ..services.sentiment_analysis import sentiment_service
from ..services.storage import storage_service

router = APIRouter(prefix="/api/calls", tags=["calls"])

# Retention period for recordings (in days)
RECORDING_RETENTION_DAYS = 3


@router.post("/upload", response_model=CallAnalysisResult)
async def upload_and_analyze_call(
    file: UploadFile = File(...),
    agent_id: Optional[str] = Form(None),
    agent_name: Optional[str] = Form(None),
    customer_name: Optional[str] = Form(None),
    customer_phone: Optional[str] = Form(None),
    call_type: CallType = Form(CallType.INCOMING),
    db: Session = Depends(get_db)
):
    """
    Upload a call recording, transcribe it using Whisper, and analyze sentiment using GPT-3.5.
    Audio is stored in Supabase Storage and automatically expires after 3 days.
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/m4a", "audio/webm", "audio/ogg"]
    if file.content_type not in allowed_types and not file.filename.endswith(('.mp3', '.wav', '.m4a', '.webm', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp3, wav, m4a, webm, ogg")
    
    # Generate unique call ID
    call_id = str(uuid.uuid4())
    
    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    
    # Save to temp file for Whisper API processing
    file_extension = os.path.splitext(file.filename)[1] or ".mp3"
    temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
    
    try:
        # Write to temp file
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(file_content)
        
        # Step 1: Upload to Supabase Storage
        storage_path, expires_at = await storage_service.upload_audio(
            file_content=file_content,
            filename=file.filename,
            call_id=call_id
        )
        
        # Step 2: Transcribe using Whisper (from temp file)
        transcription_result = await transcription_service.transcribe_audio(temp_path)
        
        # Step 3: Analyze using GPT-3.5
        analysis_result = await sentiment_service.analyze_call(transcription_result["text"])
        
        # Calculate total scores
        total_score = sum(q.score for q in analysis_result["question_scores"])
        max_score = sum(q.max_score for q in analysis_result["question_scores"])
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Create result object
        result = CallAnalysisResult(
            call_id=call_id,
            call_date=datetime.utcnow(),
            audit_date=datetime.utcnow(),
            duration_seconds=transcription_result.get("duration", 0),
            agent_id=agent_id,
            agent_name=agent_name,
            customer_name=customer_name,
            customer_phone=customer_phone,
            transcription=transcription_result["text"],
            language=transcription_result.get("language", "unknown"),
            call_summary=analysis_result["call_summary"],
            customer_sentiment=analysis_result["customer_sentiment"],
            agent_behavior=analysis_result["agent_behavior"],
            compliance_risk=analysis_result["compliance_risk"],
            question_scores=analysis_result["question_scores"],
            total_score=total_score,
            max_score=max_score,
            overall_percentage=round(overall_percentage, 2),
            customer_intent=analysis_result["customer_intent"],
            key_issues=analysis_result["key_issues"],
            resolution_status=analysis_result["resolution_status"],
            follow_up_required=analysis_result["follow_up_required"]
        )
        
        # Save to database
        db_record = CallAnalysis(
            id=call_id,
            call_date=result.call_date,
            audit_date=result.audit_date,
            duration_seconds=result.duration_seconds,
            agent_id=agent_id,
            agent_name=agent_name,
            customer_name=customer_name,
            customer_phone=customer_phone,
            transcription=result.transcription,
            language=result.language,
            call_summary=result.call_summary,
            customer_sentiment=result.customer_sentiment.model_dump(),
            agent_behavior=result.agent_behavior.model_dump(),
            compliance_risk=result.compliance_risk.model_dump(),
            question_scores=[q.model_dump() for q in result.question_scores],
            total_score=result.total_score,
            max_score=result.max_score,
            overall_percentage=result.overall_percentage,
            customer_intent=result.customer_intent,
            key_issues=result.key_issues,
            resolution_status=result.resolution_status,
            follow_up_required=result.follow_up_required,
            audio_storage_path=storage_path,
            recording_expires_at=expires_at
        )
        db.add(db_record)
        db.commit()
        
        return result
        
    except Exception as e:
        # Clean up storage on error
        try:
            if 'storage_path' in locals():
                await storage_service.delete_audio(storage_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Always clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/{call_id}", response_model=CallAnalysisResult)
async def get_call_analysis(call_id: str, db: Session = Depends(get_db)):
    """Get analysis results for a specific call"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    from ..models.schemas import CustomerSentiment, AgentBehavior, ComplianceRisk, QuestionScore
    
    return CallAnalysisResult(
        call_id=record.id,
        call_date=record.call_date,
        audit_date=record.audit_date,
        duration_seconds=record.duration_seconds,
        agent_id=record.agent_id,
        agent_name=record.agent_name,
        customer_name=record.customer_name,
        customer_phone=record.customer_phone,
        transcription=record.transcription,
        language=record.language,
        call_summary=record.call_summary,
        customer_sentiment=CustomerSentiment(**record.customer_sentiment),
        agent_behavior=AgentBehavior(**record.agent_behavior),
        compliance_risk=ComplianceRisk(**record.compliance_risk),
        question_scores=[QuestionScore(**q) for q in record.question_scores],
        total_score=record.total_score,
        max_score=record.max_score,
        overall_percentage=record.overall_percentage,
        customer_intent=record.customer_intent,
        key_issues=record.key_issues,
        resolution_status=record.resolution_status,
        follow_up_required=record.follow_up_required
    )


@router.get("/{call_id}/audio-url")
async def get_audio_url(call_id: str, db: Session = Depends(get_db)):
    """Get a signed URL for the call recording audio"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if not record.audio_storage_path:
        raise HTTPException(status_code=404, detail="No audio recording available for this call")
    
    # Check if recording has expired
    if record.recording_expires_at and record.recording_expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Recording has expired and been deleted")
    
    try:
        # Get signed URL valid for 1 hour
        signed_url = await storage_service.get_audio_url(record.audio_storage_path, expires_in=3600)
        return {
            "audio_url": signed_url,
            "expires_at": record.recording_expires_at.isoformat() if record.recording_expires_at else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio URL: {str(e)}")


@router.get("/", response_model=List[dict])
async def list_calls(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all analyzed calls"""
    records = db.query(CallAnalysis).order_by(CallAnalysis.call_date.desc()).offset(skip).limit(limit).all()
    
    return [{
        "id": r.id,
        "agent_name": r.agent_name or "Unknown",
        "customer_name": r.customer_name or "Unknown",
        "call_date": r.call_date.isoformat(),
        "duration": r.duration_seconds,
        "overall_score": r.overall_percentage,
        "sentiment": r.customer_sentiment.get("overall_sentiment", "Unknown") if r.customer_sentiment else "Unknown",
        "resolution_status": r.resolution_status,
        "has_recording": bool(r.audio_storage_path),
        "recording_expires_at": r.recording_expires_at.isoformat() if r.recording_expires_at else None
    } for r in records]


@router.delete("/{call_id}")
async def delete_call(call_id: str, db: Session = Depends(get_db)):
    """Delete a call analysis record and its associated audio recording"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Delete audio from Supabase Storage if exists
    if record.audio_storage_path:
        try:
            await storage_service.delete_audio(record.audio_storage_path)
        except Exception as e:
            print(f"Warning: Could not delete audio file: {e}")
    
    # Delete database record
    db.delete(record)
    db.commit()
    
    return {"message": "Call deleted successfully"}


@router.delete("/{call_id}/recording")
async def delete_recording_only(call_id: str, db: Session = Depends(get_db)):
    """Delete only the audio recording, keeping the analysis data"""
    record = db.query(CallAnalysis).filter(CallAnalysis.id == call_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if not record.audio_storage_path:
        raise HTTPException(status_code=404, detail="No recording to delete")
    
    # Delete from Supabase Storage
    try:
        await storage_service.delete_audio(record.audio_storage_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")
    
    # Update database record
    record.audio_storage_path = None
    record.recording_expires_at = None
    db.commit()
    
    return {"message": "Recording deleted successfully"}


@router.post("/cleanup-expired")
async def cleanup_expired_recordings(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Delete all expired recordings. This endpoint is designed to be called by a cron job.
    Protected by CRON_SECRET environment variable.
    """
    # Verify cron secret for security
    cron_secret = os.getenv("CRON_SECRET")
    if cron_secret:
        if not authorization or authorization != f"Bearer {cron_secret}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Find all records with expired recordings
    expired_records = db.query(CallAnalysis).filter(
        CallAnalysis.recording_expires_at < datetime.utcnow(),
        CallAnalysis.audio_storage_path.isnot(None)
    ).all()
    
    deleted_count = 0
    failed_count = 0
    
    for record in expired_records:
        try:
            # Delete from storage
            await storage_service.delete_audio(record.audio_storage_path)
            
            # Clear storage path in database
            record.audio_storage_path = None
            record.recording_expires_at = None
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete expired recording {record.id}: {e}")
            failed_count += 1
    
    db.commit()
    
    return {
        "message": "Cleanup completed",
        "deleted": deleted_count,
        "failed": failed_count,
        "timestamp": datetime.utcnow().isoformat()
    }
