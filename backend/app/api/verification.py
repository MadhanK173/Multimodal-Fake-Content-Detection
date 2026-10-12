import os



import tempfile



from pathlib import Path



from typing import Any, Dict, List, Tuple


from urllib.parse import urlparse







from fastapi import APIRouter, File, Form, HTTPException, UploadFile







from ..services.web_verification_service import (



    WebVerificationService,



)



from ..plugins.plugin_manager import PluginManager 







web_verifier = WebVerificationService()



plugin_manager = PluginManager()















# ============================================================



# ROUTER



# ============================================================







router = APIRouter(



    prefix="/api/v1/verify",



    tags=["web-verification"],



)











# ============================================================



# WEB VERIFICATION SERVICE



# ============================================================







web_verifier = WebVerificationService()











# ============================================================



# TEXT VERIFICATION



# ============================================================







@router.post("/text")



async def verify_text(



    text: str = Form(...)



) -> Dict[str, Any]:



    """



    TEXT VERIFICATION PIPELINE







        Input Text



            ↓



        Plugin Manager



            ↓



        Text Plugin



            ↓



        RoBERTa Prediction



            ↓



        Live Tavily Web Search



            ↓



        Evidence Verification



            ↓



        Final Prediction



    """







    text = (text or "").strip()







    if not text:



        raise HTTPException(



            status_code=400,



            detail="Text input cannot be empty.",



        )







    try:







        # ====================================================



        # STEP 1 — GET TEXT PLUGIN



        # ====================================================







        text_plugin = plugin_manager.detect_plugin(



            "text"



        )











        # ====================================================



        # STEP 2 — RUN COMPLETE TEXT PLUGIN



        # ====================================================







        plugin_result = text_plugin.analyze(



            text



        )











        # ====================================================



        # STEP 3 — GET TRAINED MODEL RESULT



        # ====================================================







        model_result = plugin_result[



            "model_result"



        ]











        # ====================================================



        # STEP 4 — GET WEB VERIFICATION RESULT



        # ====================================================







        raw_web_result = plugin_result[



            "web_verification"



        ]











        # ====================================================



        # STEP 5 — NORMALIZE WEB RESULT



        # ====================================================







        web_result = _make_public_web_result(



            raw_web_result



        )











        # ====================================================



        # STEP 6 — FINAL MODEL + WEB DECISION



        # ====================================================







        final_decision = _combine_model_and_web(



            model_result=model_result,



            web_result=web_result,



        )











        # ====================================================



        # FINAL RESPONSE



        # ====================================================







        return {



            **model_result,







            "plugin":



                plugin_result.get(



                    "plugin",



                    "text"



                ),







            "modality":



                plugin_result.get(



                    "modality",



                    "text"



                ),







            "trained_model_prediction":



                final_decision[



                    "model_prediction"



                ],







            "trained_model_prediction_raw":



                final_decision[



                    "model_prediction_raw"



                ],







            "trained_model_confidence_percent":



                final_decision[



                    "model_confidence_percent"



                ],







            "web_evidence_found":



                final_decision[



                    "evidence_found"



                ],







            "final_prediction":



                final_decision[



                    "final_prediction"



                ],







            "final_decision":



                final_decision,







            "verification_enabled":



                bool(



                    web_result.get(



                        "enabled",



                        False



                    )



                ),







            "web_verification":



                web_result,







            "verification_text":



                text,







            "processing_pipeline":



                plugin_result.get(



                    "processing_pipeline",



                    [



                        "Text Input",



                        "RoBERTa",



                        "Live Tavily Web Search",



                        "Evidence Verification",



                    ]



                ),







            "status":



                plugin_result.get(



                    "status",



                    "success"



                ),



        }











    except HTTPException:



        raise











    except Exception as error:







        raise HTTPException(



            status_code=500,



            detail=(



                "Text verification failed: "



                f"{error}"



            ),



        )







# ============================================================



# AUDIO VERIFICATION



# ============================================================







@router.post("/audio")



async def verify_audio(



    file: UploadFile = File(...)



) -> Dict[str, Any]:



    """



    AUDIO VERIFICATION PIPELINE







        Audio File



            ↓



        Whisper



            ↓



        Transcript



            ↓



        RoBERTa



            ↓



        Live Web Search



            ↓



        Evidence Verification



            ↓



        Final Prediction







    Only the extracted transcript is sent to web verification.



    The original audio file is never sent to web search.



    """







    if not file.filename:







        raise HTTPException(



            status_code=400,



            detail="Audio file is required.",



        )











    # ========================================================



    # VALIDATE FILE EXTENSION



    # ========================================================







    suffix = (



        Path(file.filename)



        .suffix



        .lower()



    )











    allowed_extensions = {



        ".mp3",



        ".wav",



        ".m4a",



        ".ogg",



        ".webm",



        ".flac",



        ".aac",



    }











    if suffix not in allowed_extensions:







        raise HTTPException(



            status_code=400,



            detail=(



                "Unsupported audio format. "



                "Use MP3, WAV, M4A, OGG, WEBM, FLAC, or AAC."



            ),



        )











    temporary_path = None











    try:







        # ====================================================



        # STEP 1 — SAVE AUDIO TEMPORARILY



        # ====================================================







        with tempfile.NamedTemporaryFile(



            delete=False,



            suffix=suffix,



        ) as temporary_file:







            content = await file.read()











            if not content:







                raise HTTPException(



                    status_code=400,



                    detail=(



                        "Uploaded audio file is empty."



                    ),



                )











            temporary_file.write(



                content



            )











            temporary_path = (



                temporary_file.name



            )











        # ====================================================



        # STEP 2 — EXISTING AUDIO PIPELINE



        #



        # Audio



        #   ↓



        # Whisper



        #   ↓



        # Transcript



        #   ↓



        # RoBERTa



        # ====================================================







                # ====================================================

        # STEP 2 — AUDIO PLUGIN

        # ====================================================



        audio_plugin = plugin_manager.detect_plugin(

            "audio"

        )



        plugin_result = audio_plugin.analyze(

            temporary_path

        )



        model_result = plugin_result["model_result"]



        transcript = plugin_result["transcript"]



        raw_web_result = plugin_result[

            "web_verification"

        ]





        # ====================================================

        # STEP 3 — VALIDATE TRANSCRIPT

        # ====================================================



        if not transcript:



            raise HTTPException(

                status_code=422,

                detail=(

                    "No usable transcript was extracted "

                    "from the audio."

                ),

            )





        # ====================================================

        # STEP 4 — WEB VERIFICATION RESULT

        # ====================================================



        # STEP 5 — NORMALIZE WEB RESULT



        # ====================================================







        web_result = _make_public_web_result(



            raw_web_result



        )











        # ====================================================



        # STEP 6 — FINAL DECISION



        # ====================================================







        final_decision = _combine_model_and_web(



            model_result=model_result,



            web_result=web_result,



        )











        # ====================================================



        # FINAL RESPONSE



        # ====================================================







        return {



            **model_result,







            "trained_model_prediction":



                final_decision[



                    "model_prediction"



                ],







            "trained_model_prediction_raw":



                final_decision[



                    "model_prediction_raw"



                ],







            "trained_model_confidence_percent":



                final_decision[



                    "model_confidence_percent"



                ],







            "web_evidence_found":



                final_decision[



                    "evidence_found"



                ],







            "final_prediction":



                final_decision[



                    "final_prediction"



                ],







            "final_decision":



                final_decision,







            "verification_enabled":



                bool(



                    web_result.get(



                        "enabled",



                        False



                    )



                ),







            "web_verification":



                web_result,







            "verification_text":



                transcript,







            "processing_pipeline": [



                "Whisper Speech-to-Text",



                "RoBERTa",



                "Live Web Search",



                "Evidence Verification",



            ],







            "original_audio_filename":



                file.filename,



        }











    except HTTPException:



        raise











    except Exception as error:







        raise HTTPException(



            status_code=500,



            detail=(



                "Audio verification failed: "



                f"{error}"



            ),



        )











    finally:







        # ====================================================



        # DELETE TEMPORARY FILE



        # ====================================================







        if (



            temporary_path



            and os.path.exists(



                temporary_path



            )



        ):







            try:







                os.remove(



                    temporary_path



                )







            except OSError:



                pass







# ============================================================



# IMAGE VERIFICATION



# ============================================================







@router.post("/image")



async def verify_image(



    file: UploadFile = File(...)



) -> Dict[str, Any]:



    """



    IMAGE VERIFICATION PIPELINE







        Image



          ↓



        OCR + Visual Analysis



          ↓



        CASE 1 / CASE 2 / CASE 3



          ↓



        RoBERTa / ResNet50 / Fusion



          ↓



        OCR Text (when available)



          ↓



        Live Tavily Web Search



          ↓



        Evidence Sources







    IMPORTANT:



        Tavily evidence is shown separately and DOES NOT override



        the trained image authenticity prediction.



    """







    if not file.filename:



        raise HTTPException(



            status_code=400,



            detail="Image file is required.",



        )







    suffix = Path(file.filename).suffix.lower()







    allowed_extensions = {



        ".jpg",



        ".jpeg",



        ".png",



        ".bmp",



        ".webp",



        ".tif",



        ".tiff",



    }







    if suffix not in allowed_extensions:



        raise HTTPException(



            status_code=400,



            detail=(



                "Unsupported image format. "



                "Use JPG, JPEG, PNG, BMP, WEBP, TIF, or TIFF."



            ),



        )







    temporary_path = None







    try:



        # ====================================================



        # STEP 1 — SAVE IMAGE TEMPORARILY



        # ====================================================



        with tempfile.NamedTemporaryFile(



            delete=False,



            suffix=suffix,



        ) as temporary_file:



            content = await file.read()







            if not content:



                raise HTTPException(



                    status_code=400,



                    detail="Uploaded image file is empty.",



                )







            temporary_file.write(content)



            temporary_path = temporary_file.name







        # ====================================================



        # STEP 2 — EXISTING IMAGE MODEL PIPELINE



        # OCR + ResNet50 + optional RoBERTa/fusion



        # ====================================================



                # ====================================================

        # STEP 2 — IMAGE PLUGIN

        # ====================================================



        image_plugin = plugin_manager.detect_plugin(

            "image"

        )



        plugin_result = image_plugin.analyze(

            temporary_path

        )



        model_result = plugin_result[

            "model_result"

        ]







        image_analysis_details = _extract_image_analysis_details(



            model_result



        )







        # ====================================================



        # STEP 3 — TRAINED IMAGE PREDICTION



        # ====================================================



        raw_prediction, confidence_percent = _extract_model_prediction(



            model_result



        )







        normalized_prediction = _normalize_model_prediction(



            raw_prediction



        )







        if normalized_prediction == "REAL":



            final_prediction = "REAL"



            display_prediction = "REAL IMAGE"



        elif normalized_prediction == "FAKE":



            final_prediction = "FAKE"



            display_prediction = "AI-GENERATED IMAGE"



        else:



            final_prediction = "UNVERIFIED"



            display_prediction = "UNVERIFIED IMAGE"







        # ====================================================



        # STEP 4 — EXTRACT OCR TEXT



        # ====================================================



        ocr_text = str(



            image_analysis_details.get("ocr_text") or ""



        ).strip()







        # ====================================================



        # STEP 5 — LIVE TAVILY WEB VERIFICATION



        #



        # ONLY OCR TEXT IS SENT TO TAVILY.



        # The image file itself is NOT sent to web search.



        # ====================================================



        if ocr_text:



            raw_web_result = web_verifier.verify_text(



                text=ocr_text,



                model_result={



                    "prediction": final_prediction,



                    "confidence_percent": confidence_percent,



                },



                modality="image_ocr_text",



            )







            web_result = _make_public_web_result(



                raw_web_result



            )







            web_sources = web_result.get("sources") or []



            web_evidence_found = bool(web_sources)







        else:



            web_result = _empty_public_result(



                enabled=False,



                verdict="NOT_APPLICABLE",



                label="NO OCR TEXT TO VERIFY",



                explanation=(



                    "No readable OCR text was detected in the image, "



                    "so there was no text claim to send to Tavily."



                ),



            )



            web_evidence_found = False







        # ====================================================



        # STEP 6 — FINAL RESPONSE



        #



        # The trained image prediction remains authoritative for



        # image authenticity. Web evidence is displayed separately.



        # ====================================================



        return {



            **model_result,







            "image_type": image_analysis_details["image_type"],



            "analysis_case": image_analysis_details["analysis_case"],



            "fusion_method": image_analysis_details["fusion_method"],



            "ocr_text": ocr_text,



            "text_model_result": image_analysis_details["text_model_result"],



            "visual_model_result": image_analysis_details["visual_model_result"],







            "trained_model_prediction": final_prediction,



            "trained_model_prediction_raw": raw_prediction,



            "trained_model_confidence_percent": confidence_percent,







            "final_prediction": final_prediction,



            "display_prediction": display_prediction,







            "web_evidence_found": web_evidence_found,



            "verification_enabled": bool(



                web_result.get("enabled", False)



            ),



            "verification_text": ocr_text,



            "web_verification": web_result,







            "processing_pipeline": [



                "OCR",



                "Visual Complexity Analysis",



                "Image Case Detection",



                "RoBERTa (when text is detected)",



                "ResNet50 (when visual content is detected)",



                "Multimodal Fusion (when applicable)",



                "OCR Text → Live Tavily Web Search (when text is detected)",



                "Evidence Verification",



            ],







            "original_image_filename": file.filename,



        }







    except HTTPException:



        raise







    except Exception as error:



        raise HTTPException(



            status_code=500,



            detail=f"Image verification failed: {error}",



        )







    finally:



        if temporary_path and os.path.exists(temporary_path):



            try:



                os.remove(temporary_path)



            except OSError:



                pass







# ============================================================



# FINAL MODEL + WEB DECISION



# ============================================================







def _combine_model_and_web(



    *,



    model_result: Dict[str, Any],



    web_result: Dict[str, Any],



) -> Dict[str, Any]:



    """



    FINAL DECISION RULES







    TEXT / AUDIO:



        Supporting web evidence



            → REAL







        Contradicting web evidence



            → FAKE







        No decisive web evidence



            → Keep trained model prediction











    IMAGE:



        The trained image model determines image authenticity.







        Web verification MUST NOT override the image result.



    """







    # ========================================================



    # EXTRACT MODEL PREDICTION



    # ========================================================







    raw_prediction, confidence_percent = (



        _extract_model_prediction(



            model_result



        )



    )











    normalized_prediction = (



        _normalize_model_prediction(



            raw_prediction



        )



    )











    # ========================================================



    # DETECT MODALITY



    # ========================================================







    modality = str(



        model_result.get(



            "modality",



            ""



        )



        or ""



    ).strip().lower()











    # ========================================================



    # WEB VERIFICATION STATUS



    # ========================================================







    verification_enabled = bool(



        web_result.get(



            "enabled",



            False



        )



    )











    supporting_count = _safe_count(



        web_result.get(



            "supporting_evidence_count"



        ),



        fallback=web_result.get(



            "exact_supporting_sources"



        ),



    )











    contradicting_count = _safe_count(



        web_result.get(



            "contradicting_evidence_count"



        ),



        fallback=web_result.get(



            "contradicting_sources"



        ),



    )











    supporting_evidence = (



        verification_enabled



        and supporting_count > 0



    )











    contradicting_evidence = (



        verification_enabled



        and contradicting_count > 0



    )











    # ========================================================



    # IMAGE MODALITY



    #



    # Image model always determines authenticity.



    # ========================================================







    if modality == "image":







        if normalized_prediction == "FAKE":







            image_prediction = (



                "AI_GENERATED_IMAGE"



            )











        elif normalized_prediction == "REAL":







            image_prediction = (



                "REAL_IMAGE"



            )











        else:







            image_prediction = (



                "UNVERIFIED"



            )











        return {



            "model_prediction":



                image_prediction,







            "model_prediction_raw":



                raw_prediction



                or "",







            "model_confidence_percent":



                confidence_percent,







            "evidence_found":



                False,







            "supporting_evidence_count":



                0,







            "contradicting_evidence_count":



                0,







            "web_verification_enabled":



                False,







            "web_verdict":



                "NOT_APPLICABLE",







            "decision_case":



                "IMAGE_MODEL_AUTHENTICITY_ONLY",







            "final_prediction":



                image_prediction,



        }











    # ========================================================



    # TEXT / AUDIO — SUPPORTING EVIDENCE



    # ========================================================







    if (



        supporting_evidence



        and supporting_count > contradicting_count



    ):







        final_prediction = "REAL"







        decision_case = (



            "SUPPORTING_EVIDENCE"



        )











    # ========================================================



    # TEXT / AUDIO — CONTRADICTING EVIDENCE



    # ========================================================







    elif (



        contradicting_evidence



        and contradicting_count > supporting_count



    ):







        final_prediction = "FAKE"







        decision_case = (



            "CONTRADICTING_EVIDENCE"



        )











    # ========================================================



    # NO DECISIVE WEB EVIDENCE



    # ========================================================







    else:







        final_prediction = (



            normalized_prediction



            or "UNVERIFIED"



        )







        decision_case = (



            "MODEL_RESULT_ONLY"



        )











    # ========================================================



    # FINAL RESPONSE



    # ========================================================







    return {



        "model_prediction":



            normalized_prediction



            or "UNVERIFIED",







        "model_prediction_raw":



            raw_prediction



            or "",







        "model_confidence_percent":



            confidence_percent,







        "evidence_found":



            supporting_evidence



            or contradicting_evidence,







        "supporting_evidence_count":



            supporting_count,







        "contradicting_evidence_count":



            contradicting_count,







        "web_verification_enabled":



            verification_enabled,







        "web_verdict":



            str(



                web_result.get(



                    "web_verdict"



                )



                or "UNVERIFIED"



            ),







        "decision_case":



            decision_case,







        "final_prediction":



            final_prediction,



    }







# ============================================================



# EXTRACT MODEL PREDICTION



# ============================================================







def _extract_model_prediction(



    model_result: Any,



) -> Tuple[str, Any]:



    """



    Supports multiple possible result structures:







        prediction







        classification.prediction







        result.prediction







        result.classification.prediction



    """







    if not isinstance(



        model_result,



        dict



    ):



        return "", None











    containers: List[Dict[str, Any]] = []











    # --------------------------------------------------------



    # TOP LEVEL



    # --------------------------------------------------------







    containers.append(



        model_result



    )











    # --------------------------------------------------------



    # TOP LEVEL CLASSIFICATION



    # --------------------------------------------------------







    classification = (



        model_result.get(



            "classification"



        )



    )











    if isinstance(



        classification,



        dict



    ):







        containers.append(



            classification



        )











    # --------------------------------------------------------



    # RESULT WRAPPER



    # --------------------------------------------------------







    nested_result = (



        model_result.get(



            "result"



        )



    )











    if isinstance(



        nested_result,



        dict



    ):







        containers.append(



            nested_result



        )











        nested_classification = (



            nested_result.get(



                "classification"



            )



        )











        if isinstance(



            nested_classification,



            dict



        ):







            containers.append(



                nested_classification



            )











    # --------------------------------------------------------



    # FIND PREDICTION



    # --------------------------------------------------------







    prediction = ""











    confidence_percent = None











    for container in containers:







        if not prediction:







            value = container.get(



                "prediction"



            )







            if value is not None:







                prediction = str(



                    value



                )











        if confidence_percent is None:







            value = container.get(



                "confidence_percent"



            )







            if isinstance(



                value,



                (int, float)



            ):







                confidence_percent = float(



                    value



                )











        if confidence_percent is None:







            value = container.get(



                "confidence"



            )







            if isinstance(



                value,



                (int, float)



            ):







                numeric_value = float(



                    value



                )











                if numeric_value <= 1:







                    confidence_percent = (



                        numeric_value * 100



                    )







                else:







                    confidence_percent = (



                        numeric_value



                    )











    return (



        prediction,



        confidence_percent,



    )











# ============================================================



# NORMALIZE MODEL PREDICTION



# ============================================================







def _normalize_model_prediction(



    prediction: str,



) -> str:







    value = (



        str(



            prediction



            or ""



        )



        .strip()



        .lower()



    )











    # ========================================================



    # REAL LABELS



    # ========================================================







    if value in {



        "label_0",



        "0",



        "real",



        "true",



        "genuine",



        "authentic",



        "real image",



        "real content",



    }:







        return "REAL"











    # ========================================================



    # FAKE / AI-GENERATED LABELS



    # ========================================================







    if value in {



        "label_1",



        "1",



        "fake",



        "false",



        "misleading",



        "fabricated",



        "ai",



        "ai-generated",



        "ai generated",



        "artificial",



        "synthetic",



        "fake image",



        "ai-generated image",



    }:







        return "FAKE"











    # ========================================================



    # PARTIAL MATCHING



    # ========================================================







    if "real" in value:



        return "REAL"











    if (



        "fake" in value



        or "ai" in value



        or "generated" in value



        or "synthetic" in value



        or "artificial" in value



    ):



        return "FAKE"











    return ""







# ============================================================



# PUBLIC WEB RESULT



# ============================================================







def _make_public_web_result(



    raw: Any,



) -> Dict[str, Any]:



    """



    Convert internal web-verifier output into the public



    frontend-safe format.







    Only source title and URL are exposed.







    No article content, snippets, or raw search data are returned.



    """







    if not isinstance(



        raw,



        dict



    ):







        return _empty_public_result(



            enabled=False,



            verdict="UNVERIFIED",



            label="NO SUCH RELATED SOURCE FOUND",



            explanation=(



                "Web verification did not return "



                "a usable result."



            ),



        )











    # ========================================================



    # CHECK WHETHER WEB VERIFICATION IS ENABLED



    # ========================================================







    enabled = bool(



        raw.get(



            "enabled",



            False



        )



    )











    if not enabled:







        return _empty_public_result(



            enabled=False,







            verdict=str(



                raw.get(



                    "web_verdict"



                )



                or "NOT_CONFIGURED"



            ),







            label=str(



                raw.get(



                    "web_verdict_label"



                )



                or "LIVE VERIFICATION NOT CONFIGURED"



            ),







            explanation=str(



                raw.get(



                    "web_explanation"



                )



                or (



                    "Live web verification "



                    "is not configured."



                )



            ),



        )











    # ========================================================



    # GET RAW EVIDENCE COUNTS



    # ========================================================







    supporting_count = _safe_count(



        raw.get(



            "supporting_evidence_count"



        ),



        fallback=raw.get(



            "exact_supporting_sources"



        ),



    )











    if supporting_count == 0:







        supporting_count = _safe_count(



            raw.get(



                "supporting_sources"



            )



        )











    contradicting_count = _safe_count(



        raw.get(



            "contradicting_evidence_count"



        ),



        fallback=raw.get(



            "contradicting_sources"



        ),



    )











    # ========================================================



    # SOURCE CANDIDATES



    # ========================================================







    supporting_candidates = raw.get("supporting_sources_list") or []



    contradicting_candidates = raw.get("contradicting_sources_list") or []



    fallback_candidates = raw.get("sources") or []







    supporting_sources = _clean_source_links(supporting_candidates)



    contradicting_sources = _clean_source_links(contradicting_candidates)



    fallback_sources = _clean_source_links(fallback_candidates)







    # ========================================================



    # DETERMINE PUBLIC VERDICT



    # ========================================================







    if supporting_count > 0 and contradicting_count == 0:

        # Expose only sources actually classified as supporting.
        # Do not silently replace them with unrelated/related sources.
        sources = supporting_sources



        if sources:



            verdict = "LIKELY_REAL"



            label = "SUPPORTING EVIDENCE FOUND"



            explanation = str(



                raw.get("web_explanation")



                or (



                    "Live sources were found that directly support "



                    "the submitted claim."



                )



            )



        else:



            verdict = "UNVERIFIED"



            label = "NO DECISIVE WEB EVIDENCE"



            explanation = (



                "Supporting evidence was reported, but no valid "



                "source link was available."



            )







    elif contradicting_count > 0 and supporting_count == 0:

        # Expose only sources actually classified as contradicting.
        sources = contradicting_sources



        verdict = "LIKELY_FALSE"



        label = "CONTRADICTING EVIDENCE FOUND"



        explanation = str(



            raw.get("web_explanation")



            or (



                "Live sources were found that contradict "



                "the submitted claim."



            )



        )







    elif supporting_count > 0 and contradicting_count > 0:



        sources = _clean_source_links(



            supporting_sources + contradicting_sources



        )



        if not sources:



            sources = fallback_sources



        verdict = "QUESTIONABLE"



        label = "CONFLICTING EVIDENCE FOUND"



        explanation = str(



            raw.get("web_explanation")



            or (



                "Live sources contain both supporting and "



                "contradicting evidence."



            )



        )







    else:



        sources = fallback_sources



        verdict = "UNVERIFIED"



        label = "NO DECISIVE WEB EVIDENCE"



        explanation = str(



            raw.get("web_explanation")



            or (



                "No live source directly confirms or contradicts "



                "the submitted claim."



            )



        )







    # ========================================================



    # FINAL PUBLIC WEB RESULT



    # ========================================================







    return {



        "enabled": True,



        "status": str(raw.get("status") or "success"),



        "verification_modality": str(



            raw.get("verification_modality") or "text"



        ),



        "web_verdict": verdict,



        "web_verdict_label": label,



        "web_explanation": explanation,



        "related_sources_found": len(sources),



        "supporting_sources": len(supporting_sources),



        "exact_supporting_sources": len(supporting_sources),



        "supporting_evidence_count": supporting_count,



        "contradicting_sources": len(contradicting_sources),



        "contradicting_evidence_count": contradicting_count,



        "supporting_sources_list": supporting_sources,



        "contradicting_sources_list": contradicting_sources,



        "sources": sources,



    }







# ============================================================



# CLEAN SOURCE LINKS



# ============================================================







def _clean_source_links(
    value: Any,
) -> List[Dict[str, str]]:
    """
    Keep only safe, clickable source objects.

    Every public source must contain:
        {
            "title": "...",
            "url": "https://..."
        }

    The previous implementation validated against the literal string
    "https\\://", which can reject normal URLs such as
    "https://www.reuters.com/...". That caused the backend to report
    evidence counts while the frontend received an empty sources list.

    This function validates the URL structurally and preserves the
    original URL returned by Tavily.
    """

    if not isinstance(value, list):
        return []

    output: List[Dict[str, str]] = []
    seen = set()

    for item in value:
        if not isinstance(item, dict):
            continue

        title = str(
            item.get("title")
            or item.get("name")
            or item.get("domain")
            or "Source"
        ).strip()

        url = str(
            item.get("url")
            or item.get("link")
            or item.get("href")
            or ""
        ).strip()

        if not url:
            continue

        try:
            parsed = urlparse(url)
        except Exception:
            continue

        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
        ):
            continue

        key = url.rstrip("/").lower()

        if key in seen:
            continue

        seen.add(key)

        output.append(
            {
                "title": title[:300] or "Source",
                "url": url,
            }
        )

        if len(output) >= 6:
            break

    return output



# ============================================================

# EMPTY PUBLIC RESULT



# ============================================================







def _empty_public_result(



    *,



    enabled: bool,



    verdict: str,



    label: str,



    explanation: str,



) -> Dict[str, Any]:







    return {



        "enabled":



            enabled,







        "status":



            (



                "success"



                if enabled



                else "not_configured"



            ),







        "verification_modality":



            "text",







        "web_verdict":



            verdict,







        "web_verdict_label":



            label,







        "web_explanation":



            explanation,







        "related_sources_found":



            0,







        "supporting_sources":



            0,







        "exact_supporting_sources":



            0,







        "supporting_evidence_count":



            0,







        "contradicting_sources":



            0,







        "contradicting_evidence_count":



            0,







        "sources":



            [],



    }











# ============================================================



# SAFE COUNT



# ============================================================







def _safe_count(



    value: Any,



    fallback: Any = 0,



) -> int:



    """



    Safely convert evidence values into counts.







    Supports:







        int



        float



        list



        tuple



        set



        dict







    This prevents failures when the web service returns



    either a numeric count or an actual list of sources.



    """







    candidate = value











    if candidate is None:







        candidate = fallback











    if candidate is None:



        return 0











    # --------------------------------------------------------



    # COLLECTION



    # --------------------------------------------------------







    if isinstance(



        candidate,



        (



            list,



            tuple,



            set,



            dict,



        )



    ):







        return max(



            0,



            len(candidate)



        )











    # --------------------------------------------------------



    # NUMBER / STRING



    # --------------------------------------------------------







    try:







        return max(



            0,



            int(candidate)



        )







    except (



        TypeError,



        ValueError,



    ):







        return 0



# ============================================================



# IMAGE ANALYSIS DETAILS EXTRACTION



# ============================================================







def _extract_image_analysis_details(



    result: Any,



) -> Dict[str, Any]:



    """



    Extract image analysis details from the multimodal image



    prediction result.







    Supports:







        - Text Only



        - Image Only



        - Text + Image







    Also extracts:







        - analysis_case



        - fusion_method



        - OCR text



        - text model result



        - visual model result



    """







    if not isinstance(result, dict):







        return {



            "image_type": "Image Only",



            "analysis_case": "",



            "fusion_method": "",



            "ocr_text": "",



            "text_model_result": None,



            "visual_model_result": None,



        }











    # ========================================================



    # GET NESTED RESULT IF AVAILABLE



    # ========================================================







    nested_result = result.get("result")







    if not isinstance(nested_result, dict):



        nested_result = {}











    # ========================================================



    # HELPER FUNCTION



    # ========================================================







    def get_value(key: str):







        return (



            result.get(key)



            if result.get(key) is not None



            else nested_result.get(key)



        )











    # ========================================================



    # EXTRACT OCR TEXT



    # ========================================================







    ocr_text = _extract_ocr_text(result)











    # ========================================================



    # EXTRACT ANALYSIS CASE



    # ========================================================







    analysis_case = (



        get_value("analysis_case")



        or get_value("case")



        or get_value("image_case")



        or ""



    )











    # ========================================================



    # EXTRACT FUSION METHOD



    # ========================================================







    fusion_method = (



        get_value("fusion_method")



        or get_value("fusion")



        or ""



    )











    # ========================================================



    # EXTRACT TEXT MODEL RESULT



    # ========================================================







    text_model_result = (



        get_value("text_model_result")



        or get_value("text_result")



        or None



    )











    # ========================================================



    # EXTRACT VISUAL MODEL RESULT



    # ========================================================







    visual_model_result = (



        get_value("visual_model_result")



        or get_value("image_model_result")



        or get_value("visual_result")



        or None



    )











    # ========================================================



    # DETERMINE IMAGE TYPE



    # ========================================================







    normalized_case = str(



        analysis_case



        or ""



    ).strip().lower()











    # CASE 1 → TEXT ONLY



    if (



        "text_only" in normalized_case



        or "text only" in normalized_case



        or normalized_case == "case_1"



        or normalized_case == "case1"



    ):







        image_type = "Text Only"











    # CASE 2 → IMAGE ONLY



    elif (



        "image_only" in normalized_case



        or "image only" in normalized_case



        or normalized_case == "case_2"



        or normalized_case == "case2"



    ):







        image_type = "Image Only"











    # CASE 3 → TEXT + IMAGE



    elif (



        "text_image" in normalized_case



        or "text+image" in normalized_case



        or "text_image_fusion" in normalized_case



        or normalized_case == "case_3"



        or normalized_case == "case3"



        or "multimodal" in normalized_case



    ):







        image_type = "Text + Image"











    # ========================================================



    # FALLBACK DETECTION



    # ========================================================







    else:







        has_ocr = bool(



            str(ocr_text or "").strip()



        )











        has_text_model = isinstance(



            text_model_result,



            dict



        )











        has_visual_model = isinstance(



            visual_model_result,



            dict



        )











        # Text + Image



        if has_ocr and has_text_model and has_visual_model:







            image_type = "Text + Image"











        # Text Only



        elif has_ocr and has_text_model and not has_visual_model:







            image_type = "Text Only"











        # Image Only



        else:







            image_type = "Image Only"











    # ========================================================



    # RETURN DETAILS



    # ========================================================







    return {



        "image_type": image_type,







        "analysis_case": analysis_case,







        "fusion_method": fusion_method,







        "ocr_text": ocr_text,







        "text_model_result": text_model_result,







        "visual_model_result": visual_model_result,



    }







# ============================================================



# OCR TEXT EXTRACTION



# ============================================================







def _extract_ocr_text(



    result: Any,



) -> str:



    """



    Extract OCR text from multiple possible image result layouts.







    Supported structures:







        {



            "ocr_text": "..."



        }







        {



            "result": {



                "ocr_text": "..."



            }



        }



    """







    if not isinstance(



        result,



        dict



    ):



        return ""











    # ========================================================



    # HELPER



    # ========================================================







    def clean(



        value: Any



    ) -> str:







        if isinstance(



            value,



            str



        ):







            value = value.strip()







            if value:



                return value







        return ""











    # ========================================================



    # TOP LEVEL OCR TEXT



    # ========================================================







    ocr_text = clean(



        result.get(



            "ocr_text"



        )



    )











    if ocr_text:



        return ocr_text











    # ========================================================



    # RESULT WRAPPER



    # ========================================================







    nested_result = (



        result.get(



            "result"



        )



    )











    if isinstance(



        nested_result,



        dict



    ):







        ocr_text = clean(



            nested_result.get(



                "ocr_text"



            )



        )











        if ocr_text:



            return ocr_text











    return ""







# ============================================================



# TRANSCRIPT EXTRACTION



# ============================================================







def _extract_transcript(



    result: Any,



) -> str:



    """



    Extract transcript from multiple possible audio result layouts.







    Supported examples:







        {



            "transcript": "..."



        }







        {



            "classification": {



                "transcript": "..."



            }



        }







        {



            "result": {



                "transcript": "..."



            }



        }







        {



            "result": {



                "classification": {



                    "transcript": "..."



                }



            }



        }



    """







    if not isinstance(



        result,



        dict



    ):



        return ""











    # ========================================================



    # HELPER



    # ========================================================







    def clean(



        value: Any



    ) -> str:







        if isinstance(



            value,



            str



        ):







            value = value.strip()







            if value:



                return value







        return ""











    # ========================================================



    # TOP LEVEL TRANSCRIPT



    # ========================================================







    for key in (



        "transcript",



        "text",



        "transcription",



        "transcribed_text",



        "speech_text",



    ):







        transcript = clean(



            result.get(key)



        )







        if transcript:



            return transcript











    # ========================================================



    # TOP LEVEL CLASSIFICATION



    # ========================================================







    classification = (



        result.get(



            "classification"



        )



    )











    if isinstance(



        classification,



        dict



    ):







        for key in (



            "transcript",



            "text",



            "transcription",



            "transcribed_text",



            "speech_text",



        ):







            transcript = clean(



                classification.get(key)



            )







            if transcript:



                return transcript











    # ========================================================



    # RESULT WRAPPER



    # ========================================================







    nested_result = (



        result.get(



            "result"



        )



    )











    if isinstance(



        nested_result,



        dict



    ):







        for key in (



            "transcript",



            "text",



            "transcription",



            "transcribed_text",



            "speech_text",



        ):







            transcript = clean(



                nested_result.get(key)



            )







            if transcript:



                return transcript











        # ====================================================



        # RESULT.CLASSIFICATION



        # ====================================================







        nested_classification = (



            nested_result.get(



                "classification"



            )



        )











        if isinstance(



            nested_classification,



            dict



        ):







            for key in (



                "transcript",



                "text",



                "transcription",



                "transcribed_text",



                "speech_text",



            ):







                transcript = clean(



                    nested_classification.get(key)



                )







                if transcript:



                    return transcript











    return ""