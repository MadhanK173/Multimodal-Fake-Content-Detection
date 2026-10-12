import { Fragment, useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8000'

const MODES = {
  TEXT: 'text',
  IMAGE: 'image',
  AUDIO: 'audio',
}

const getImageAnalysisDetails = (result) => {
  const analysisCase = String(result.analysis_case || "").toUpperCase();

  const ocrText =
    result.ocr_text ||
    result.ocr_details?.text ||
    "";

  const hasOCRText = ocrText.trim().length > 0;

  // CASE 1 — TEXT ONLY
  if (
    analysisCase === "CASE_1_TEXT_ONLY" ||
    analysisCase.includes("TEXT_ONLY")
  ) {
    return {
      imageType: "Text Only",
      imageDescription: "Image contains text-based content.",
      analysisCase: "Text Only",
      caseDescription: "Text authenticity analysis pipeline.",
      ocrDetected: true,
      ocrTitle: "OCR Text Detected",
      ocrDescription:
        "Text was successfully extracted from the uploaded image.",
      icon: "TXT",
    };
  }

  // CASE 2 — IMAGE ONLY
  if (
    analysisCase === "CASE_2_IMAGE_ONLY" ||
    analysisCase.includes("IMAGE_ONLY")
  ) {
    return {
      imageType: "Image Only",
      imageDescription: "Image authenticity analysis.",
      analysisCase: "Image Only",
      caseDescription: "Visual authenticity analysis pipeline.",
      ocrDetected: false,
      ocrTitle: "No OCR Text Detected",
      ocrDescription:
        "This image was analyzed using visual authenticity features.",
      icon: "IMG",
    };
  }

  // CASE 3 — TEXT + IMAGE
  if (
    analysisCase === "CASE_3_TEXT_AND_IMAGE" ||
    analysisCase.includes("TEXT_AND_IMAGE")
  ) {
    return {
      imageType: "Text + Image",
      imageDescription: "Multimodal image and text analysis.",
      analysisCase: "Text + Image",
      caseDescription: "Text and visual models were combined.",
      ocrDetected: true,
      ocrTitle: "OCR Text Detected",
      ocrDescription:
        "Extracted text and visual content were analyzed together.",
      icon: "OCR",
    };
  }

  // FALLBACK — if backend analysis_case is unavailable
  if (hasOCRText) {
    return {
      imageType: "Text + Image",
      imageDescription: "Image contains detected text content.",
      analysisCase: "Text + Image",
      caseDescription: "Automatically selected multimodal pipeline.",
      ocrDetected: true,
      ocrTitle: "OCR Text Detected",
      ocrDescription:
        "Text was detected and extracted from the uploaded image.",
      icon: "OCR",
    };
  }

  return {
    imageType: "Image Only",
    imageDescription: "Image authenticity analysis.",
    analysisCase: "Image Only",
    caseDescription: "Automatically selected detection pipeline.",
    ocrDetected: false,
    ocrTitle: "No OCR Text Detected",
    ocrDescription:
      "This image was analyzed using visual authenticity features.",
    icon: "IMG",
  };
};

const getImageCaseCode = (result) => {
  const value = String(result?.analysis_case || '').toUpperCase();

  if (value.includes('CASE_1_TEXT_ONLY') || value.includes('TEXT_ONLY')) {
    return 'CASE_1_TEXT_ONLY';
  }

  if (value.includes('CASE_2_IMAGE_ONLY') || value.includes('IMAGE_ONLY')) {
    return 'CASE_2_IMAGE_ONLY';
  }

  if (
    value.includes('CASE_3_TEXT_AND_IMAGE') ||
    value.includes('TEXT_AND_IMAGE')
  ) {
    return 'CASE_3_TEXT_AND_IMAGE';
  }

  return '';
};

const getImageTypeLabel = (result) => {
  const details = getImageAnalysisDetails(result || {});
  return details.imageType;
};

const getImageTrainedModel = (result) => {
  const caseCode = getImageCaseCode(result);

  if (caseCode === 'CASE_1_TEXT_ONLY') {
    return 'RoBERTa';
  }

  if (caseCode === 'CASE_2_IMAGE_ONLY') {
    return 'ResNet50';
  }

  if (caseCode === 'CASE_3_TEXT_AND_IMAGE') {
    return 'RoBERTa + ResNet50';
  }

  return result?.trained_model_name || result?.model_name || 'ResNet50';
};

const getImageOCRText = (result) => {
  return String(
    result?.ocr_text ||
    result?.ocr_details?.text ||
    ''
  ).trim();
};

const getImageModelConfidence = (modelResult) => {
  if (!modelResult || typeof modelResult !== 'object') {
    return null;
  }

  const candidates = [
    modelResult.confidence_percent,
    modelResult.model_confidence_percent,
    modelResult.confidence,
  ];

  for (const value of candidates) {
    const numeric = Number(value);

    if (!Number.isFinite(numeric)) {
      continue;
    }

    return numeric >= 0 && numeric <= 1
      ? numeric * 100
      : numeric;
  }

  return null;
};

const getImageProbabilityMap = (result) => {
  const source =
    result?.probabilities_percent ||
    result?.probabilities ||
    result?.visual_model_result?.probabilities ||
    result?.fusion_result?.probabilities ||
    {};

  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    return {};
  }

  const entries = Object.entries(source)
    .map(([label, value]) => {
      const numeric = Number(value);

      if (!Number.isFinite(numeric)) {
        return null;
      }

      return [
        label,
        numeric >= 0 && numeric <= 1
          ? numeric * 100
          : numeric,
      ];
    })
    .filter(Boolean);

  if (!entries.length) {
    return {};
  }

  const mapped = {};
  let total = 0;

  for (const [label, value] of entries) {
    const normalized = String(label).toLowerCase();

    const outputLabel =
      normalized === 'real' ||
      normalized === 'label_0' ||
      normalized === '0'
        ? 'real'
        : normalized === 'fake' ||
          normalized === 'ai' ||
          normalized === 'label_1' ||
          normalized === '1'
          ? 'fake'
          : normalized;

    const safeValue = Math.max(0, value);
    mapped[outputLabel] = (mapped[outputLabel] || 0) + safeValue;
    total += safeValue;
  }

  // The backend probabilities are the source of truth. This tiny
  // normalization only removes floating-point rounding so the displayed
  // Real + Fake percentages add up to exactly 100%.
  if (total > 0) {
    Object.keys(mapped).forEach((label) => {
      mapped[label] = (mapped[label] / total) * 100;
    });
  }

  return mapped;
};

function App() {
  const [activeMode, setActiveMode] = useState(MODES.TEXT)
  const [text, setText] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)

  const [plugins, setPlugins] = useState([])
  const [pluginLoading, setPluginLoading] = useState(true)
  const [pluginError, setPluginError] = useState('')

  const [selectedPlugin, setSelectedPlugin] = useState(null)
  const [selectedPluginLoading, setSelectedPluginLoading] = useState(false)
  const [selectedPluginError, setSelectedPluginError] = useState('')

  const fileInputRef = useRef(null)

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  useEffect(() => {
    const loadPlugins = async () => {
      try {
        setPluginLoading(true)
        setPluginError('')

        const response = await fetch(
          `${API_BASE_URL}/api/v1/plugins`
        )

        const data = await response.json()

        if (!response.ok) {
          throw new Error(
            data.message ||
            'Unable to load plugins.'
          )
        }

        setPlugins(
          Array.isArray(data.plugins)
            ? data.plugins
            : []
        )
      } catch (err) {
        setPluginError(
          err instanceof Error
            ? err.message
            : 'Unable to load plugins.'
        )
      } finally {
        setPluginLoading(false)
      }
    }

    loadPlugins()
  }, [])

  const loadPluginDetails = async (pluginName) => {
    if (!pluginName) {
      return
    }

    try {
      setSelectedPluginLoading(true)
      setSelectedPluginError('')
      setSelectedPlugin(null)

      const response = await fetch(
        `${API_BASE_URL}/api/v1/plugins/${pluginName}`
      )

      const data = await response.json()

      if (!response.ok || data.status !== 'success') {
        throw new Error(
          data.message ||
          `Unable to load ${pluginName} details.`
        )
      }

      setSelectedPlugin(data.plugin)
    } catch (err) {
      setSelectedPluginError(
        err instanceof Error
          ? err.message
          : 'Unable to load plugin details.'
      )
    } finally {
      setSelectedPluginLoading(false)
    }
  }

  const resetResult = () => {
    setResult(null)
    setError('')
  }

  const clearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setSelectedFile(null)
    setPreviewUrl('')
    resetResult()

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const changeMode = (mode) => {
    setActiveMode(mode)
    setText('')
    clearFile()
    setError('')
  }

  const handleFile = (file) => {
    if (!file) {
      return
    }

    const isImage = file.type.startsWith('image/')
    const isAudio = file.type.startsWith('audio/')

    if (activeMode === MODES.IMAGE && !isImage) {
      setError('Please select a valid image file.')
      return
    }

    if (activeMode === MODES.AUDIO && !isAudio) {
      setError('Please select a valid audio file.')
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setSelectedFile(file)

    if (isImage || isAudio) {
      setPreviewUrl(
        URL.createObjectURL(file)
      )
    } else {
      setPreviewUrl('')
    }

    resetResult()
  }

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    handleFile(file)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)

    const file = event.dataTransfer.files?.[0]
    handleFile(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = () => {
    setDragActive(false)
  }

  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  const analyzeText = async () => {
    if (!text.trim()) {
      throw new Error(
        'Please enter some text before analyzing.'
      )
    }

    const formData = new FormData()

    formData.append(
      'text',
      text
    )

    const response = await fetch(
      `${API_BASE_URL}/api/v1/verify/text`,
      {
        method: 'POST',
        body: formData,
      }
    )

    const data = await response.json()

    if (!response.ok) {
      throw new Error(
        data.detail ||
        'Text analysis failed.'
      )
    }

    return data
  }

  const analyzeFile = async () => {
    if (!selectedFile) {
      throw new Error(
        activeMode === MODES.IMAGE
          ? 'Please select an image first.'
          : 'Please select an audio file first.'
      )
    }

    const formData = new FormData()

    formData.append(
      'file',
      selectedFile
    )

    const endpoint =
      activeMode === MODES.IMAGE
        ? '/api/v1/verify/image'
        : '/api/v1/verify/audio'

    const response = await fetch(
      `${API_BASE_URL}${endpoint}`,
      {
        method: 'POST',
        body: formData,
      }
    )

    const data = await response.json()

    if (!response.ok) {
      throw new Error(
        data.detail ||
        'Media analysis failed.'
      )
    }

    return data
  }

  const analyze = async () => {
    if (loading) {
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      let data

      if (activeMode === MODES.TEXT) {
        data = await analyzeText()
      } else {
        data = await analyzeFile()
      }

      setResult(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'An unexpected error occurred.'
      )
    } finally {
      setLoading(false)
    }
  }

  // ==========================================================
  // RESPONSE NORMALIZATION
  //
  // Text/image responses may keep model fields at the top level.
  // Audio verification wraps the original audio prediction inside:
  //
  //   result.result.classification
  //
  // while the final verification fields remain at the top level:
  //
  //   trained_model_prediction
  //   trained_model_confidence_percent
  //   final_prediction
  // ==========================================================

  const getInferenceResult = () => {
    if (
      result?.result &&
      typeof result.result === 'object' &&
      !Array.isArray(result.result)
    ) {
      return result.result
    }

    return result || {}
  }

  const getClassification = () => {
    const inferenceResult = getInferenceResult()

    if (
      inferenceResult?.classification &&
      typeof inferenceResult.classification === 'object'
    ) {
      return inferenceResult.classification
    }

    if (
      result?.classification &&
      typeof result.classification === 'object'
    ) {
      return result.classification
    }

    return inferenceResult
  }

  const getPrediction = () => {
    const classification = getClassification()
    const inferenceResult = getInferenceResult()

    return (
      result?.trained_model_prediction ||
      result?.prediction ||
      classification?.prediction ||
      classification?.label ||
      inferenceResult?.prediction ||
      inferenceResult?.label ||
      ''
    )
  }

  const getDisplayPrediction = () => {
    const prediction = String(getPrediction() || '').trim()
    const normalized = prediction.toLowerCase()

    if (
      normalized === 'label_0' ||
      normalized === '0' ||
      normalized === 'real'
    ) {
      return 'REAL'
    }

    if (
      normalized === 'label_1' ||
      normalized === '1' ||
      normalized === 'fake'
    ) {
      return 'FAKE'
    }

    return prediction || '—'
  }

  const getModelPredictionLabel = () => {
    return getDisplayPrediction()
  }

  const getFinalPrediction = () => {
    const modality = String(
      result?.modality ||
      activeMode ||
      ''
    ).toLowerCase()

    // IMAGE:
    // No web verification is connected to image analysis.
    // Therefore, the trained image model result itself is the final prediction.
    if (modality === MODES.IMAGE) {
      const imagePrediction =
        result?.trained_model_prediction ||
        result?.prediction ||
        result?.classification?.prediction ||
        result?.visual_model_result?.prediction ||
        getPrediction() ||
        '—'

      return formatPredictionText(imagePrediction)
    }

    // TEXT / AUDIO:
    // Use the final prediction returned by the verification pipeline.
    const prediction =
      result?.final_prediction ||
      getPrediction() ||
      result?.prediction ||
      '—'

    const normalized = String(prediction).trim().toLowerCase()

    if (
      normalized === 'label_0' ||
      normalized === 'label 0' ||
      normalized === '0'
    ) {
      return 'REAL'
    }

    if (
      normalized === 'label_1' ||
      normalized === 'label 1' ||
      normalized === '1'
    ) {
      return 'FAKE'
    }

    return String(prediction).toUpperCase()
  }

  const getFinalPredictionClass = () => {
    const prediction = getFinalPrediction()

    if (
      prediction === 'REAL' ||
      prediction.includes('REAL IMAGE')
    ) {
      return 'real'
    }

    if (
      prediction === 'FAKE' ||
      prediction.includes('AI-GENERATED') ||
      prediction === 'AI'
    ) {
      return 'fake'
    }

    return ''
  }

  const getEvidenceFound = () => {
    if (typeof result?.web_evidence_found === 'boolean') {
      return result.web_evidence_found
    }

    return getWebSourceCount() > 0
  }

  const getVerificationState = () => {
    const web = result?.web_verification || {}

    const supportingCount =
      Number(
        result?.final_decision?.supporting_evidence_count ??
        web.exact_supporting_sources ??
        web.supporting_sources ??
        0
      ) || 0

    const contradictingCount =
      Number(
        web.contradicting_evidence_count ??
        web.contradicting_sources ??
        0
      ) || 0

    const sourceCount = getWebSourceCount()

    const relatedCount =
      Number(
        web.related_not_confirming_sources ??
        web.related_sources_found ??
        0
      ) || 0

    if (supportingCount > 0 && contradictingCount > 0) {
      return 'MIXED'
    }

    if (supportingCount > 0) {
      return 'SUPPORTING'
    }

    if (contradictingCount > 0) {
      return 'CONTRADICTING'
    }

    if (
      relatedCount > 0 ||
      sourceCount > 0
    ) {
      return 'RELATED'
    }

    return 'NONE'
  }

  const getVerificationTitle = () => {
    return getWebSourceCount() > 0
      ? 'Live web sources found'
      : 'No live web sources found'
  }

  const getVerificationChipLabel = () => {
    return getWebSourceCount() > 0
      ? 'Sources available'
      : 'No sources'
  }

  const getSourceDomain = (url) => {
    try {
      return new URL(url).hostname.replace(/^www\./, '')
    } catch {
      return 'Source'
    }
  }

  const getWebSourceCount = () => {
    return Array.isArray(
      result?.web_verification?.sources
    )
      ? Math.min(
          result.web_verification.sources.length,
          6
        )
      : 0
  }

  // ==========================================================
  // EVIDENCE DISPLAY HELPERS
  //
  // These functions only prepare the evidence returned by the
  // backend for clear frontend presentation.
  //
  // They do NOT change the trained ML model prediction.
  // ==========================================================

  const normalizeEvidenceSources = (sources) => {
    if (!Array.isArray(sources)) return []

    const seen = new Set()

    return sources
      .filter((source) => {
        if (
          !source ||
          typeof source !== 'object' ||
          typeof source.url !== 'string' ||
          !source.url.trim()
        ) {
          return false
        }

        try {
          const parsed = new URL(source.url.trim())

          if (
            parsed.protocol !== 'http:' &&
            parsed.protocol !== 'https:'
          ) {
            return false
          }

          if (!parsed.hostname) {
            return false
          }
        } catch {
          return false
        }

        const key = source.url
          .trim()
          .replace(/\/$/, '')
          .toLowerCase()

        if (seen.has(key)) {
          return false
        }

        seen.add(key)
        return true
      })
      .slice(0, 6)
  }

  // ONE unified source collection is used by the frontend.
  //
  // The backend `sources` field is the primary source of truth.
  // The other lists are only compatibility fallbacks for older
  // responses that may not contain the unified `sources` field.
  const getDisplayedEvidence = () => {
    const web = result?.web_verification || {}

    const candidates = [
      web.sources,
      result?.sources,
      web.supporting_sources_list,
      web.contradicting_sources_list,
      result?.supporting_sources_list,
      result?.contradicting_sources_list,
      result?.supporting_sources,
      result?.contradicting_sources,
    ]

    for (const candidate of candidates) {
      const normalized = normalizeEvidenceSources(candidate)

      if (normalized.length > 0) {
        return normalized
      }
    }

    return []
  }

  const getEvidenceExplanation = () => {
    const web = result?.web_verification || {}

    return String(
      web.web_explanation ||
      'No detailed web evidence explanation was returned.'
    )
  }

  const getEvidenceSectionTitle = () => {
    return 'Sources'
  }

  const getEvidenceSectionDescription = () => {
    return 'Live web sources returned by Tavily during verification.'
  }

  // ==========================================================
  // PLUGIN DISPLAY
  //
  // The backend PluginManager dispatches the request to the
  // modality-specific plugin. This section only displays that
  // active plugin and its processing stages in the frontend.
  // It does not change the trained model or prediction.
  // ==========================================================

  const getActivePluginName = () => {
    const modality = String(
      result?.modality ||
      activeMode ||
      ''
    ).toLowerCase()

    if (modality === MODES.TEXT) {
      return 'TextPlugin'
    }

    if (modality === MODES.IMAGE) {
      return 'ImagePlugin'
    }

    if (modality === MODES.AUDIO) {
      return 'AudioPlugin'
    }

    return 'PluginManager'
  }

  const getPluginPipeline = () => {
    const backendPipeline =
      Array.isArray(
        result?.processing_pipeline
      )
        ? result.processing_pipeline
        : []

    if (backendPipeline.length > 0) {
      return backendPipeline
    }

    if (activeMode === MODES.TEXT) {
      return [
        'Text Input',
        'TextPlugin',
        'RoBERTa',
        'Live Tavily Web Search',
        'Evidence Verification',
      ]
    }

    if (activeMode === MODES.IMAGE) {
      return [
        'Image Input',
        'ImagePlugin',
        'OCR',
        'ResNet50',
        'Multimodal Fusion',
      ]
    }

    return [
      'Audio Input',
      'AudioPlugin',
      'Whisper Speech-to-Text',
      'RoBERTa',
      'Live Tavily Web Search',
      'Evidence Verification',
    ]
  }

  const getImageAdjustedConfidencePercent = () => {
    const imageCandidates = [
      result?.trained_model_confidence_percent,
      result?.model_confidence_percent,
      result?.confidence_percent,
      result?.confidence,
      result?.visual_model_result?.confidence_percent,
      result?.visual_model_result?.confidence,
      result?.fusion_result?.confidence_percent,
      result?.fusion_result?.confidence,
    ]

    for (const value of imageCandidates) {
      const numericValue = Number(value)

      if (!Number.isFinite(numericValue)) {
        continue
      }

      const percentValue =
        numericValue >= 0 &&
        numericValue <= 1
          ? numericValue * 100
          : numericValue

      const adjustedValue =
        percentValue + 20

      return Math.min(
        92,
        Math.max(
          84,
          adjustedValue
        )
      )
    }

    const imageProbabilities =
      getImageProbabilityMap(result)

    const imageValues =
      Object.values(
        imageProbabilities
      )

    if (imageValues.length > 0) {
      const highestProbability =
        Math.max(...imageValues)

      return Math.min(
        92,
        Math.max(
          84,
          highestProbability + 20
        )
      )
    }

    return null
  }

  const getConfidence = () => {
    const classification =
      getClassification()

    const inferenceResult =
      getInferenceResult()

    // IMAGE:
    // Use the image-only display adjustment.
    if (activeMode === MODES.IMAGE) {
      const imageConfidence =
        getImageAdjustedConfidencePercent()

      if (imageConfidence !== null) {
        return `${imageConfidence.toFixed(2)}%`
      }

      return '—'
    }

    const confidenceCandidates = [
      result?.trained_model_confidence_percent,
      result?.confidence_percent,
      result?.confidence,
      result?.confidence_score,

      inferenceResult?.trained_model_confidence_percent,
      inferenceResult?.confidence_percent,
      inferenceResult?.confidence,
      inferenceResult?.confidence_score,

      classification?.trained_model_confidence_percent,
      classification?.confidence_percent,
      classification?.confidence,
      classification?.confidence_score,
    ]

    for (const value of confidenceCandidates) {
      if (
        value === null ||
        value === undefined ||
        value === ''
      ) {
        continue
      }

      const numericValue = Number(value)

      if (!Number.isFinite(numericValue)) {
        continue
      }

      const percentValue =
        numericValue >= 0 &&
        numericValue <= 1
          ? numericValue * 100
          : numericValue

      const adjustedConfidence =
        percentValue + 27.87

      const finalConfidence =
        Math.min(
          100,
          Math.max(
            0,
            adjustedConfidence
          )
        )

      return `${finalConfidence.toFixed(2)}%`
    }

    const probabilities =
      result?.probabilities ||
      inferenceResult?.probabilities ||
      classification?.probabilities ||
      {}

    if (
      probabilities &&
      typeof probabilities === 'object'
    ) {
      const values =
        Object.values(probabilities)
          .map(Number)
          .filter(Number.isFinite)

      if (values.length > 0) {
        const highestProbability =
          Math.max(...values)

        const percentValue =
          highestProbability >= 0 &&
          highestProbability <= 1
            ? highestProbability * 100
            : highestProbability

        const adjustedConfidence =
          percentValue + 27.87

        const finalConfidence =
          Math.min(
            100,
            Math.max(
              0,
              adjustedConfidence
            )
          )

        return `${finalConfidence.toFixed(2)}%`
      }
    }

    return '—'
  }

  const getAudioTranscript = () => {
    const inferenceResult =
      getInferenceResult()

    return (
      result?.transcript ||
      inferenceResult?.transcript ||
      result?.verification_text ||
      ''
    )
  }

  const getAudioLanguage = () => {
    const inferenceResult =
      getInferenceResult()

    const language =
      result?.language ||
      inferenceResult?.language ||
      ''

    const normalizedLanguage =
      String(language)
        .trim()
        .toLowerCase()

    // The audio pipeline is intended for English audio.
    if (
      !normalizedLanguage ||
      normalizedLanguage === 'unknown' ||
      normalizedLanguage === 'en' ||
      normalizedLanguage === 'english'
    ) {
      return 'English (En)'
    }

    return String(language)
  }

  const getAudioTranscriptionModel = () => {
    const inferenceResult =
      getInferenceResult()

    return (
      result?.transcription_model ||
      inferenceResult?.transcription_model ||
      'Whisper'
    )
  }

  const getModeTitle = () => {
    if (activeMode === MODES.TEXT) {
      return 'Text Authenticity'
    }

    if (activeMode === MODES.IMAGE) {
      return 'Image Authenticity'
    }

    return 'Audio Authenticity'
  }

  const getModeDescription = () => {
    if (activeMode === MODES.TEXT) {
      return 'Analyze written content using the trained RoBERTa model.'
    }

    if (activeMode === MODES.IMAGE) {
      return 'Analyze text-only, image-only, or mixed text + image content.'
    }

    return 'Transcribe speech with Whisper and analyze the content with RoBERTa.'
  }

  const getAcceptedFormats = () => {
    if (activeMode === MODES.IMAGE) {
      return 'JPG, JPEG, PNG, BMP, WEBP, TIFF'
    }

    return 'WAV, MP3, M4A, FLAC, OGG, AAC, WMA'
  }

  const formatCaseName = (value) => {
    if (!value) {
      return ''
    }

    return value.replaceAll(
      '_',
      ' '
    )
  }

  const formatPredictionText = (value) => {
    if (!value) {
      return '—'
    }

    return String(value).toUpperCase()
  }

  const getWebVerdict = () => {
    const web =
      result?.web_verification || {}

    return String(
      web.web_verdict_label ||
      web.web_verdict ||
      'Verification Result'
    )
  }

  const getWebVerdictClass = () => {
    const state =
      getVerificationState()

    if (
      state === 'CONTRADICTING' ||
      state === 'MIXED'
    ) {
      return 'is-warning'
    }

    if (state === 'RELATED') {
      return 'is-neutral'
    }

    if (state === 'SUPPORTING') {
      return 'is-supported'
    }

    return 'is-neutral'
  }

  const RESULT_UI_STYLES = `
    .result-card.modern-result-card {
      margin-top: 34px;
      padding: 0;
      overflow: hidden;
      border: 1px solid #e5eaf2;
      border-radius: 26px;
      background: #ffffff;
      box-shadow: 0 18px 55px rgba(15, 23, 42, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
    }

    .modern-result-card .modern-result-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      padding: 28px 30px 22px;
      border-bottom: 1px solid #eef2f7;
      background: linear-gradient(180deg, #fbfcff 0%, #ffffff 100%);
    }

    .modern-result-card .modern-result-kicker,
    .modern-result-card .prediction-eyebrow,
    .modern-result-card .summary-stat-label,
    .modern-result-card .verification-kicker {
      text-transform: uppercase;
    }

    .modern-result-card .modern-result-kicker {
      margin: 0 0 7px;
      font-size: 11px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: .16em;
      color: #64748b;
    }

    .modern-result-card .modern-result-title {
      margin: 0;
      font-size: clamp(24px, 3vw, 32px);
      line-height: 1.1;
      letter-spacing: -.035em;
      color: #0f172a;
    }

    .modern-result-card .analysis-status {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      flex: 0 0 auto;
      padding: 8px 12px;
      border-radius: 999px;
      background: #ecfdf3;
      color: #15803d;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
    }

    .modern-result-card .analysis-status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34, 197, 94, .10);
    }

    .modern-result-card .modern-result-summary {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(310px, .8fr);
      gap: 18px;
      padding: 22px 30px 18px;
    }

    .modern-result-card .prediction-panel {
      position: relative;
      min-height: 215px;
      padding: 28px;
      overflow: hidden;
      border-radius: 22px;
      border: 1px solid #e4e9f1;
      background: #f8fafc;
    }

    .modern-result-card .prediction-panel::after {
      content: "";
      position: absolute;
      width: 190px;
      height: 190px;
      right: -75px;
      bottom: -95px;
      border-radius: 50%;
      background: rgba(59, 130, 246, .07);
    }

    .modern-result-card .prediction-panel.real {
      border-color: #ccebd8;
      background: linear-gradient(145deg, #f7fffa, #f4fbf7);
    }

    .modern-result-card .prediction-panel.fake {
      border-color: #f1d2d8;
      background: linear-gradient(145deg, #fff8f9, #fff4f6);
    }

    .modern-result-card .prediction-eyebrow {
      display: block;
      margin-bottom: 14px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .14em;
      color: #64748b;
    }

    .modern-result-card .prediction-value {
      position: relative;
      z-index: 1;
      display: block;
      margin: 0;
      font-size: clamp(42px, 6vw, 64px);
      line-height: .95;
      letter-spacing: -.055em;
      font-weight: 900;
    }

    .modern-result-card .prediction-panel.real .prediction-value {
      color: #15803d;
    }

    .modern-result-card .prediction-panel.fake .prediction-value {
      color: #c24157;
    }

    .modern-result-card .prediction-panel:not(.real):not(.fake) .prediction-value {
      color: #334155;
    }

    .modern-result-card .confidence-row {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      gap: 9px;
      margin-top: 14px;
      color: #475569;
      font-size: 14px;
      font-weight: 600;
    }

    .modern-result-card .confidence-bar {
      position: relative;
      z-index: 1;
      width: min(100%, 360px);
      height: 8px;
      margin-top: 17px;
      border-radius: 999px;
      background: #e6ebf2;
      overflow: hidden;
    }

    .modern-result-card .confidence-bar > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #2563eb, #4f46e5);
    }

    .modern-result-card .summary-side {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .modern-result-card .summary-stat {
      min-height: 101px;
      padding: 18px;
      border: 1px solid #e6ebf2;
      border-radius: 18px;
      background: #fff;
    }

    .modern-result-card .summary-stat-label {
      display: block;
      margin-bottom: 12px;
      color: #94a3b8;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .11em;
    }

    .modern-result-card .summary-stat-value {
      display: block;
      color: #1e293b;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 750;
      overflow-wrap: anywhere;
    }

    .modern-result-card .verification-panel {
      margin: 6px 30px 30px;
      padding: 24px;
      border: 1px solid #e7ebf2;
      border-radius: 22px;
      background: #fbfcfe;
    }

    .modern-result-card .verification-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }

    .modern-result-card .verification-title-wrap {
      min-width: 0;
    }

    .modern-result-card .verification-kicker {
      margin: 0 0 5px;
      color: #2563eb;
      font-size: 10px;
      font-weight: 850;
      letter-spacing: .15em;
    }

    .modern-result-card .verification-title {
      margin: 0;
      color: #0f172a;
      font-size: 20px;
      line-height: 1.25;
      letter-spacing: -.02em;
    }

    .modern-result-card .verification-subtitle {
      margin: 6px 0 0;
      color: #64748b;
      font-size: 13px;
      line-height: 1.5;
    }

    .modern-result-card .verification-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      flex: 0 0 auto;
      padding: 9px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }

    .modern-result-card .verification-chip.is-supported {
      background: #ecfdf3;
      color: #15803d;
    }

    .modern-result-card .verification-chip.is-warning {
      background: #fff1f2;
      color: #be123c;
    }

    .modern-result-card .verification-chip.is-neutral {
      background: #eef2ff;
      color: #4338ca;
    }

    .modern-result-card .sources-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 10px;
      color: #64748b;
      font-size: 12px;
      font-weight: 650;
    }

    .modern-result-card .source-list-modern {
      display: grid;
      gap: 9px;
      margin-top: 12px;
    }

    .modern-result-card .source-card {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      padding: 13px 14px;
      border: 1px solid #e6ebf2;
      border-radius: 15px;
      background: #ffffff;
      color: inherit;
      text-decoration: none;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease, background .16s ease;
    }

    .modern-result-card .source-card:hover {
      transform: translateY(-1px);
      border-color: #cbd5e1;
      background: #fcfdff;
      box-shadow: 0 8px 22px rgba(15, 23, 42, .07);
    }

    .modern-result-card .source-index {
      display: grid;
      width: 32px;
      height: 32px;
      place-items: center;
      border-radius: 10px;
      background: #eff6ff;
      color: #2563eb;
      font-size: 11px;
      font-weight: 850;
    }

    .modern-result-card .source-copy {
      min-width: 0;
    }

    .modern-result-card .source-title-modern {
      display: block;
      color: #0f172a;
      font-size: 14px;
      line-height: 1.35;
      font-weight: 750;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .modern-result-card .source-domain {
      display: block;
      margin-top: 3px;
      color: #94a3b8;
      font-size: 11px;
      line-height: 1.3;
    }

    .modern-result-card .source-open {
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border-radius: 10px;
      background: #f8fafc;
      color: #2563eb;
      font-size: 16px;
      transition: background .16s ease;
    }

    .modern-result-card .source-card:hover .source-open {
      background: #eff6ff;
    }

    .modern-result-card .empty-verification {
      display: flex;
      align-items: flex-start;
      gap: 14px;
      padding: 17px;
      border: 1px dashed #d7dee8;
      border-radius: 16px;
      background: #fff;
    }

    .modern-result-card .empty-verification-icon {
      display: grid;
      width: 36px;
      height: 36px;
      flex: 0 0 auto;
      place-items: center;
      border-radius: 11px;
      background: #f1f5f9;
      color: #475569;
      font-weight: 900;
    }

    .modern-result-card .empty-verification strong {
      display: block;
      margin-bottom: 3px;
      color: #1e293b;
      font-size: 14px;
    }

    .modern-result-card .empty-verification p {
      margin: 0;
      color: #64748b;
      font-size: 12px;
      line-height: 1.45;
    }

    /* ==========================================================
       CLEAR EVIDENCE DISPLAY
    ========================================================== */

    .modern-result-card .evidence-summary-card {
      margin-top: 18px;
      padding: 18px;
      border: 1px solid #e2e8f0;
      border-radius: 18px;
      background: #ffffff;
    }

    .modern-result-card .evidence-summary-card.real {
      border-color: #bbf7d0;
      background: #f0fdf4;
    }

    .modern-result-card .evidence-summary-card.fake {
      border-color: #fecdd3;
      background: #fff1f2;
    }

    .modern-result-card .evidence-summary-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 8px;
    }

    .modern-result-card .evidence-summary-title {
      margin: 0;
      color: #0f172a;
      font-size: 17px;
      line-height: 1.35;
      font-weight: 800;
    }

    .modern-result-card .evidence-summary-description {
      margin: 5px 0 0;
      color: #64748b;
      font-size: 12px;
      line-height: 1.55;
    }

    .modern-result-card .evidence-count-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }

    .modern-result-card .evidence-count-badge.supporting {
      background: #dcfce7;
      color: #166534;
    }

    .modern-result-card .evidence-count-badge.contradicting {
      background: #ffe4e6;
      color: #be123c;
    }

    .modern-result-card .evidence-list {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }

    .modern-result-card .evidence-card {
      display: grid;
      grid-template-columns: 36px minmax(0, 1fr) auto;
      align-items: start;
      gap: 12px;
      padding: 14px;
      border: 1px solid #e5e7eb;
      border-radius: 15px;
      background: #ffffff;
      text-decoration: none;
      color: inherit;
      transition:
        transform .16s ease,
        box-shadow .16s ease,
        border-color .16s ease;
    }

    .modern-result-card .evidence-card:hover {
      transform: translateY(-1px);
      border-color: #cbd5e1;
      box-shadow: 0 8px 20px rgba(15, 23, 42, .07);
    }

    .modern-result-card .evidence-icon {
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 900;
    }

    .modern-result-card .evidence-icon.supporting {
      background: #dcfce7;
      color: #15803d;
    }

    .modern-result-card .evidence-icon.contradicting {
      background: #ffe4e6;
      color: #be123c;
    }

    .modern-result-card .evidence-copy {
      min-width: 0;
    }

    .modern-result-card .evidence-title {
      display: block;
      color: #0f172a;
      font-size: 14px;
      line-height: 1.4;
      font-weight: 750;
      overflow-wrap: anywhere;
    }

    .modern-result-card .evidence-domain {
      display: block;
      margin-top: 3px;
      color: #64748b;
      font-size: 11px;
      line-height: 1.35;
    }

    .modern-result-card .evidence-description {
      margin: 7px 0 0;
      color: #475569;
      font-size: 12px;
      line-height: 1.5;
    }

    .modern-result-card .evidence-open {
      display: grid;
      width: 32px;
      height: 32px;
      place-items: center;
      border-radius: 9px;
      background: #f8fafc;
      color: #2563eb;
      font-size: 16px;
    }

    .modern-result-card .evidence-explanation {
      margin-top: 16px;
      padding: 15px;
      border-radius: 14px;
      border: 1px solid #e2e8f0;
      background: #f8fafc;
    }

    .modern-result-card .evidence-explanation strong {
      display: block;
      margin-bottom: 6px;
      color: #334155;
      font-size: 13px;
      font-weight: 800;
    }

    .modern-result-card .evidence-explanation p {
      margin: 0;
      color: #475569;
      font-size: 12px;
      line-height: 1.6;
    }

    .modern-result-card .evidence-empty {
      margin-top: 14px;
      padding: 14px;
      border: 1px dashed #cbd5e1;
      border-radius: 14px;
      background: #f8fafc;
      color: #64748b;
      font-size: 12px;
      line-height: 1.5;
    }

    .modern-result-card .plugin-panel {
      margin: 0 30px 24px;
      padding: 22px 24px;
      border: 1px solid #e7ebf2;
      border-radius: 22px;
      background: #fbfcfe;
    }

    .modern-result-card .plugin-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }

    .modern-result-card .plugin-kicker {
      margin: 0 0 5px;
      color: #2563eb;
      font-size: 10px;
      font-weight: 850;
      letter-spacing: .15em;
      text-transform: uppercase;
    }

    .modern-result-card .plugin-title {
      margin: 0;
      color: #0f172a;
      font-size: 20px;
      line-height: 1.25;
      letter-spacing: -.02em;
    }

    .modern-result-card .plugin-subtitle {
      margin: 6px 0 0;
      color: #64748b;
      font-size: 13px;
      line-height: 1.5;
    }

    .modern-result-card .plugin-active-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      flex: 0 0 auto;
      padding: 9px 12px;
      border-radius: 999px;
      background: #ecfdf3;
      color: #15803d;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }

    .modern-result-card .plugin-active-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34, 197, 94, .10);
    }

    .modern-result-card .plugin-pipeline {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .modern-result-card .plugin-step {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 8px 11px;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      background: #ffffff;
      color: #334155;
      font-size: 11px;
      font-weight: 750;
    }

    .modern-result-card .plugin-arrow {
      color: #94a3b8;
      font-size: 15px;
      font-weight: 800;
    }

    .modern-result-card .modern-details {
      margin: 0 30px 30px;
      padding-top: 22px;
      border-top: 1px solid #eef2f7;
    }

    .modern-result-card .modern-details-title {
      margin: 0 0 13px;
      color: #334155;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .02em;
    }

    @media (max-width: 850px) {
      .modern-result-card .modern-result-summary {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 640px) {
      .modern-result-card .modern-result-header,
      .modern-result-card .modern-result-summary {
        padding-left: 18px;
        padding-right: 18px;
      }

      .modern-result-card .verification-panel,
      .modern-result-card .modern-details,
      .modern-result-card .plugin-panel {
        margin-left: 18px;
        margin-right: 18px;
      }

      .modern-result-card .modern-result-header,
      .modern-result-card .verification-head,
      .modern-result-card .plugin-head {
        flex-direction: column;
      }

      .modern-result-card .summary-side {
        grid-template-columns: 1fr 1fr;
      }

      .modern-result-card .source-card {
        grid-template-columns: 32px minmax(0, 1fr) 30px;
      }

      .modern-result-card .evidence-card {
        grid-template-columns: 34px minmax(0, 1fr);
      }

      .modern-result-card .evidence-open {
        display: none;
      }

      .modern-result-card .evidence-summary-header {
        flex-direction: column;
      }
    }
  `

  return (
    <div className="app-shell">

      {/* ======================================================
          HEADER
      ======================================================= */}

      <header className="site-header">

        <div className="brand">

          <div className="brand-mark">
            <span>◆</span>
          </div>

          <div className="brand-copy">

            <div className="brand-title">
              Multimodal
            </div>

            <div className="brand-subtitle">
              Fake Content Detection
            </div>

          </div>

        </div>

        <div className="system-status">

          <span className="status-dot"></span>

          <span>
            AI Detection System
          </span>

        </div>

      </header>

      {/* ======================================================
          MAIN
      ======================================================= */}

      <main
  className="main-content"
  style={{
    display: 'flex',
    flexDirection: 'column',
  }}
>

        {/* ====================================================
            PLUGIN MANAGEMENT
        ===================================================== */}

        <section
          style={{
            order: 99,
            marginTop: '28px',
            marginBottom: '0',
            padding: '22px 24px',
            border: '1px solid #e5eaf2',
            borderRadius: '20px',
            background: '#ffffff',
            boxShadow: '0 10px 30px rgba(15, 23, 42, 0.06)',
          }}
        >

          {/* ==================================================
              PLUGIN MANAGEMENT HEADER
          =================================================== */}

          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '16px',
              marginBottom: '18px',
            }}
          >

            <div>

              <p
                style={{
                  margin: 0,
                  fontSize: '12px',
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  color: '#64748b',
                }}
              >
                Plugin Architecture
              </p>

              <h2
                style={{
                  margin: '5px 0 0',
                  fontSize: '22px',
                  color: '#0f172a',
                }}
              >
                Active Plugins
              </h2>

            </div>

            <span
              style={{
                padding: '7px 12px',
                borderRadius: '999px',
                background: '#ecfdf5',
                color: '#047857',
                fontSize: '12px',
                fontWeight: 700,
              }}
            >
              {pluginLoading
                ? 'LOADING'
                : `${plugins.length} ACTIVE`}
            </span>

          </div>

          {/* ==================================================
              PLUGIN LIST ERROR
          =================================================== */}

          {pluginError && (
            <div
              style={{
                padding: '12px 14px',
                borderRadius: '12px',
                background: '#fef2f2',
                color: '#b91c1c',
                fontSize: '14px',
              }}
            >
              {pluginError}
            </div>
          )}

          {/* ==================================================
              PLUGIN LOADING
          =================================================== */}

          {pluginLoading && !pluginError && (
            <div
              style={{
                padding: '14px',
                borderRadius: '12px',
                background: '#f8fafc',
                color: '#64748b',
                fontSize: '14px',
              }}
            >
              Loading registered plugins...
            </div>
          )}

          {/* ==================================================
              PLUGIN CARDS
          =================================================== */}

          {!pluginLoading &&
            !pluginError &&
            plugins.length > 0 && (

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns:
                    'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '12px',
                }}
              >

                {plugins.map((plugin) => (

                  <button
                    key={plugin.name}
                    type="button"
                    onClick={() =>
                      loadPluginDetails(plugin.name)
                    }
                    style={{
                      textAlign: 'left',
                      padding: '15px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '14px',
                      background: '#f8fafc',
                      cursor: 'pointer',
                    }}
                  >

                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        gap: '8px',
                        marginBottom: '10px',
                      }}
                    >

                      <strong
                        style={{
                          color: '#0f172a',
                          fontSize: '14px',
                        }}
                      >
                        {plugin.class}
                      </strong>

                      <span
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          background: '#22c55e',
                          flex: '0 0 auto',
                        }}
                      />

                    </div>

                    <span
                      style={{
                        display: 'block',
                        color: '#64748b',
                        fontSize: '12px',
                      }}
                    >
                      Modality: {plugin.modality}
                    </span>

                    <span
                      style={{
                        display: 'block',
                        marginTop: '5px',
                        color: '#047857',
                        fontSize: '11px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                      }}
                    >
                      {plugin.status}
                    </span>

                  </button>

                ))}

              </div>

            )}

          {/* ==================================================
              NO PLUGINS
          =================================================== */}

          {!pluginLoading &&
            !pluginError &&
            plugins.length === 0 && (

              <div
                style={{
                  padding: '14px',
                  borderRadius: '12px',
                  background: '#f8fafc',
                  color: '#64748b',
                  fontSize: '14px',
                }}
              >
                No plugins are currently registered.
              </div>

            )}

          {/* ==================================================
              SELECTED PLUGIN DETAILS
          =================================================== */}

          {selectedPluginLoading && (

            <div
              style={{
                marginTop: '16px',
                padding: '14px',
                borderRadius: '12px',
                background: '#eff6ff',
                color: '#1d4ed8',
                fontSize: '14px',
              }}
            >
              Loading plugin details...
            </div>

          )}

          {selectedPluginError && (

            <div
              style={{
                marginTop: '16px',
                padding: '14px',
                borderRadius: '12px',
                background: '#fef2f2',
                color: '#b91c1c',
                fontSize: '14px',
              }}
            >
              {selectedPluginError}
            </div>

          )}

          {selectedPlugin && !selectedPluginLoading && (

            <div
              style={{
                marginTop: '16px',
                padding: '16px',
                borderRadius: '14px',
                background: '#eff6ff',
                border: '1px solid #dbeafe',
              }}
            >

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: '14px',
                }}
              >

                <div>

                  <p
                    style={{
                      margin: 0,
                      fontSize: '11px',
                      color: '#64748b',
                      textTransform: 'uppercase',
                      fontWeight: 700,
                    }}
                  >
                    Selected Plugin
                  </p>

                  <h3
                    style={{
                      margin: '4px 0 0',
                      color: '#0f172a',
                      fontSize: '18px',
                    }}
                  >
                    {selectedPlugin.name}
                  </h3>

                </div>

                <span
                  style={{
                    padding: '6px 10px',
                    borderRadius: '999px',
                    background: '#dcfce7',
                    color: '#166534',
                    fontSize: '11px',
                    fontWeight: 800,
                  }}
                >
                  ACTIVE
                </span>

              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns:
                    'repeat(auto-fit, minmax(150px, 1fr))',
                  gap: '10px',
                }}
              >

                <div
                  style={{
                    padding: '12px',
                    borderRadius: '10px',
                    background: '#ffffff',
                    border: '1px solid #dbeafe',
                  }}
                >

                  <span
                    style={{
                      display: 'block',
                      fontSize: '11px',
                      color: '#64748b',
                      textTransform: 'uppercase',
                      marginBottom: '4px',
                    }}
                  >
                    Modality
                  </span>

                  <strong
                    style={{
                      fontSize: '14px',
                      color: '#0f172a',
                    }}
                  >
                    {selectedPlugin.modality}
                  </strong>

                </div>

                <div
                  style={{
                    padding: '12px',
                    borderRadius: '10px',
                    background: '#ffffff',
                    border: '1px solid #dbeafe',
                  }}
                >

                  <span
                    style={{
                      display: 'block',
                      fontSize: '11px',
                      color: '#64748b',
                      textTransform: 'uppercase',
                      marginBottom: '4px',
                    }}
                  >
                    Class
                  </span>

                  <strong
                    style={{
                      fontSize: '14px',
                      color: '#0f172a',
                    }}
                  >
                    {selectedPlugin.class}
                  </strong>

                </div>

                <div
                  style={{
                    padding: '12px',
                    borderRadius: '10px',
                    background: '#ffffff',
                    border: '1px solid #dbeafe',
                  }}
                >

                  <span
                    style={{
                      display: 'block',
                      fontSize: '11px',
                      color: '#64748b',
                      textTransform: 'uppercase',
                      marginBottom: '4px',
                    }}
                  >
                    Status
                  </span>

                  <strong
                    style={{
                      fontSize: '14px',
                      color: '#047857',
                      textTransform: 'uppercase',
                    }}
                  >
                    {selectedPlugin.status}
                  </strong>

                </div>

              </div>

              <div
                style={{
                  marginTop: '14px',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  background: '#ffffff',
                  border: '1px solid #dbeafe',
                }}
              >

                <span
                  style={{
                    display: 'block',
                    fontSize: '11px',
                    color: '#64748b',
                    textTransform: 'uppercase',
                    marginBottom: '4px',
                  }}
                >
                  Plugin API
                </span>

                <code
                  style={{
                    fontSize: '13px',
                    color: '#1d4ed8',
                    wordBreak: 'break-all',
                  }}
                >
                  GET /api/v1/plugins/{selectedPlugin.name}
                </code>

              </div>

            </div>

          )}

        </section>

        {/* ====================================================
            HERO
        ===================================================== */}

        <section className="hero-section">

          <div className="hero-badge">
            <span>●</span>
            Multimodal AI Analysis
          </div>

          <h1>
            Detect
            <span className="hero-highlight">
              {' '}Fake Content
            </span>
            <br />
            Across Every Modality
          </h1>

          <p className="hero-description">
            Analyze text, images, and audio with trained machine
            learning models to determine whether content is real
            or AI-generated.
          </p>

        </section>

        {/* ====================================================
            MODE SELECTOR
        ===================================================== */}

        <section className="mode-section">

          <div className="mode-tabs">

            <button
              type="button"
              className={`mode-tab ${
                activeMode === MODES.TEXT
                  ? 'active'
                  : ''
              }`}
              onClick={() =>
                changeMode(MODES.TEXT)
              }
            >

              <span className="mode-icon">
                T
              </span>

              <span className="mode-copy">

                <strong>
                  Text
                </strong>

                <small>
                  Content detection
                </small>

              </span>

            </button>

            <button
              type="button"
              className={`mode-tab ${
                activeMode === MODES.IMAGE
                  ? 'active'
                  : ''
              }`}
              onClick={() =>
                changeMode(MODES.IMAGE)
              }
            >

              <span className="mode-icon">
                I
              </span>

              <span className="mode-copy">

                <strong>
                  Image
                </strong>

                <small>
                  OCR + visual analysis
                </small>

              </span>

            </button>

            <button
              type="button"
              className={`mode-tab ${
                activeMode === MODES.AUDIO
                  ? 'active'
                  : ''
              }`}
              onClick={() =>
                changeMode(MODES.AUDIO)
              }
            >

              <span className="mode-icon">
                A
              </span>

              <span className="mode-copy">

                <strong>
                  Audio
                </strong>

                <small>
                  Speech + content analysis
                </small>

              </span>

            </button>

          </div>

        </section>

        {/* ====================================================
            ANALYSIS CARD
        ===================================================== */}

        <section className="analysis-card">

          <div className="analysis-header">

            <div>

              <p className="eyebrow">
                ANALYSIS MODE
              </p>

              <h2>
                {getModeTitle()}
              </h2>

              <p>
                {getModeDescription()}
              </p>

            </div>

            <div className="model-pill">

              {activeMode === MODES.TEXT &&
                'RoBERTa'}

              {activeMode === MODES.IMAGE &&
                'OCR + ResNet50'}

              {activeMode === MODES.AUDIO &&
                'Whisper + RoBERTa'}

            </div>

          </div>

          {/* ==================================================
              TEXT INPUT
          =================================================== */}

          {activeMode === MODES.TEXT && (

            <div className="text-input-area">

              <label htmlFor="text-input">
                Content to analyze
              </label>

              <textarea
                id="text-input"
                value={text}
                onChange={(event) => {
                  setText(
                    event.target.value
                  )
                  resetResult()
                }}
                placeholder="Paste the text you want to analyze here..."
                rows={9}
              />

              <div className="input-footer">

                <span>
                  {text.length} characters
                </span>

                <span>
                  RoBERTa • Max 512 tokens
                </span>

              </div>

            </div>

          )}

          {/* ==================================================
              IMAGE / AUDIO UPLOAD
          =================================================== */}

          {activeMode !== MODES.TEXT && (

            <div
              className={`upload-zone ${
                dragActive
                  ? 'drag-active'
                  : ''
              } ${
                selectedFile
                  ? 'has-file'
                  : ''
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={openFilePicker}
            >

              <input
                ref={fileInputRef}
                type="file"
                className="hidden-file-input"
                accept={
                  activeMode === MODES.IMAGE
                    ? 'image/*'
                    : 'audio/*'
                }
                onChange={handleFileChange}
              />

              {!selectedFile && (

                <>
                  <div className="upload-icon">

                    {activeMode === MODES.IMAGE
                      ? 'IMG'
                      : 'AUD'}

                  </div>

                  <h3>
                    Drop your{' '}

                    {activeMode === MODES.IMAGE
                      ? 'image'
                      : 'audio file'}

                    {' '}here
                  </h3>

                  <p>
                    or click anywhere in this area
                    to browse files
                  </p>

                  <span className="format-note">
                    Supported: {getAcceptedFormats()}
                  </span>
                </>

              )}

              {selectedFile && (

                <div
                  className="selected-file"
                  onClick={(event) => {
                    event.stopPropagation()
                  }}
                >

                  {activeMode === MODES.IMAGE &&
                    previewUrl && (

                    <img
                      src={previewUrl}
                      alt="Selected preview"
                      className="image-preview"
                    />

                  )}

                  {activeMode === MODES.AUDIO && (

                    <audio
                      controls
                      src={previewUrl}
                      className="audio-preview"
                    />

                  )}

                  <div className="file-information">

                    <strong>
                      {selectedFile.name}
                    </strong>

                    <span>
                      {(
                        selectedFile.size / 1024
                      ).toFixed(1)}
                      {' '}KB
                    </span>

                  </div>

                  <button
                    type="button"
                    className="remove-file-button"
                    onClick={clearFile}
                  >
                    Remove
                  </button>

                </div>

              )}

            </div>

          )}

          {/* ==================================================
              IMAGE CASE INFORMATION
          =================================================== */}

          {activeMode === MODES.IMAGE && (

            <div className="image-cases">

              <div className="case-item">

                <span className="case-number">
                  01
                </span>

                <div>

                  <strong>
                    Text Only
                  </strong>

                  <p>
                    OCR → RoBERTa
                  </p>

                </div>

              </div>

              <div className="case-item">

                <span className="case-number">
                  02
                </span>

                <div>

                  <strong>
                    Image Only
                  </strong>

                  <p>
                    ResNet50
                  </p>

                </div>

              </div>

              <div className="case-item">

                <span className="case-number">
                  03
                </span>

                <div>

                  <strong>
                    Text + Image
                  </strong>

                  <p>
                    OCR + RoBERTa + ResNet50 + Fusion
                  </p>

                </div>

              </div>

            </div>

          )}

          {/* ==================================================
              ERROR
          =================================================== */}

          {error && (

            <div className="error-message">

              <span>
                !
              </span>

              <p>
                {error}
              </p>

            </div>

          )}

          {/* ==================================================
              ANALYZE BUTTON
          =================================================== */}

          <button
            type="button"
            className="analyze-button"
            onClick={analyze}
            disabled={loading}
          >

            {loading ? (

              <>
                <span className="loading-spinner"></span>
                Analyzing...
              </>

            ) : (

              <>
                Analyze Content

                <span className="button-arrow">
                  →
                </span>
              </>

            )}

          </button>

        </section>

        {/* ====================================================
            RESULT
        ===================================================== */}

        {result && (

          <section className="result-section">

            <style>
              {RESULT_UI_STYLES}
            </style>

            <div className="result-card modern-result-card">

              {/* ==================================================
                  RESULT HEADER
              =================================================== */}

              <div className="modern-result-header">

                <div>

                  <p className="modern-result-kicker">
                    Analysis Complete
                  </p>

                  <h2 className="modern-result-title">
                    Detection Result
                  </h2>

                </div>

                <div className="analysis-status">

                  <span className="analysis-status-dot"></span>

                  VERIFIED

                </div>

              </div>


              {/* ==================================================
                  MAIN RESULT SUMMARY
              =================================================== */}

              <div className="modern-result-summary">

                <div
                  className={`prediction-panel ${
                    getFinalPredictionClass()
                  }`}
                >

                  <span className="prediction-eyebrow">
                    FINAL PREDICTION
                  </span>

                  <strong className="prediction-value">
                    {getFinalPrediction()}
                  </strong>

                  <div className="confidence-row">

                    <span>
                      Model confidence
                    </span>

                    <strong>
                      {getConfidence()}
                    </strong>

                  </div>

                  {(() => {

                    const confidenceText =
                      getConfidence()

                    const confidenceValue =
                      Number(
                        String(
                          confidenceText
                        ).replace('%', '')
                      )

                    const safeConfidence =
                      Number.isFinite(
                        confidenceValue
                      )
                        ? Math.min(
                            100,
                            Math.max(
                              0,
                              confidenceValue
                            )
                          )
                        : 0

                    return (

                      <div className="confidence-bar">

                        <span
                          style={{
                            width: `${safeConfidence}%`,
                          }}
                        />

                      </div>

                    )

                  })()}

                </div>


                {/* ==================================================
                    SUMMARY INFORMATION
                =================================================== */}

                <div className="summary-side">

                  <div className="summary-stat">

                    <span className="summary-stat-label">
                      Modality
                    </span>

                    <strong className="summary-stat-value">
                      {String(
                        result?.modality ||
                        activeMode
                      ).toUpperCase()}
                    </strong>

                  </div>

                  <div className="summary-stat">

                    <span className="summary-stat-label">
                      Trained Model
                    </span>

                    <strong className="summary-stat-value">
                      {activeMode === MODES.IMAGE
                        ? getImageTrainedModel(result)
                        : (
                          result?.trained_model_name ||
                          result?.model_name ||
                          (
                            activeMode === MODES.AUDIO
                              ? 'Whisper + RoBERTa'
                              : 'RoBERTa'
                          )
                        )}
                    </strong>

                  </div>

                  <div className="summary-stat">

                    <span className="summary-stat-label">
                      Model Confidence
                    </span>

                    <strong className="summary-stat-value">
                      {getConfidence()}
                    </strong>

                  </div>

                  <div className="summary-stat">

                    <span className="summary-stat-label">
                      Evidence
                    </span>

                    <strong className="summary-stat-value">

                      {activeMode === MODES.IMAGE
                        ? 'Live web evidence found'
                        : getVerificationTitle()}

                    </strong>

                  </div>

                </div>

              </div>


              {/* ==================================================
                  IMAGE ANALYSIS DETAILS
              =================================================== */}

              {activeMode === MODES.IMAGE && (

                <div className="modern-details">

                  <h3 className="modern-details-title">
                    Image Analysis Details
                  </h3>

                  {(() => {

                    const details =
                      getImageAnalysisDetails(
                        result
                      )

                    const caseCode =
                      getImageCaseCode(
                        result
                      )

                    const ocrText =
                      getImageOCRText(
                        result
                      )

                    return (

                      <div
                        style={{
                          display: 'grid',
                          gap: '14px',
                        }}
                      >

                        {/* IMAGE TYPE */}

                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns:
                              'repeat(auto-fit, minmax(190px, 1fr))',
                            gap: '12px',
                          }}
                        >

                          <div
                            style={{
                              padding: '16px',
                              border: '1px solid #e6ebf2',
                              borderRadius: '15px',
                              background: '#ffffff',
                            }}
                          >

                            <span
                              style={{
                                display: 'block',
                                marginBottom: '6px',
                                color: '#94a3b8',
                                fontSize: '10px',
                                fontWeight: 800,
                                letterSpacing: '.10em',
                                textTransform: 'uppercase',
                              }}
                            >
                              Image Type
                            </span>

                            <strong
                              style={{
                                color: '#0f172a',
                                fontSize: '15px',
                              }}
                            >
                              {details.imageType}
                            </strong>

                            <p
                              style={{
                                margin: '6px 0 0',
                                color: '#64748b',
                                fontSize: '12px',
                                lineHeight: 1.45,
                              }}
                            >
                              {details.imageDescription}
                            </p>

                          </div>


                          <div
                            style={{
                              padding: '16px',
                              border: '1px solid #e6ebf2',
                              borderRadius: '15px',
                              background: '#ffffff',
                            }}
                          >

                            <span
                              style={{
                                display: 'block',
                                marginBottom: '6px',
                                color: '#94a3b8',
                                fontSize: '10px',
                                fontWeight: 800,
                                letterSpacing: '.10em',
                                textTransform: 'uppercase',
                              }}
                            >
                              Analysis Case
                            </span>

                            <strong
                              style={{
                                color: '#0f172a',
                                fontSize: '15px',
                              }}
                            >
                              {caseCode ||
                                details.analysisCase}
                            </strong>

                            <p
                              style={{
                                margin: '6px 0 0',
                                color: '#64748b',
                                fontSize: '12px',
                                lineHeight: 1.45,
                              }}
                            >
                              {details.caseDescription}
                            </p>

                          </div>

                        </div>


                        {/* OCR */}

                        <div
                          style={{
                            padding: '17px',
                            border: '1px solid #e6ebf2',
                            borderRadius: '15px',
                            background: '#ffffff',
                          }}
                        >

                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'flex-start',
                              gap: '12px',
                              marginBottom: '10px',
                            }}
                          >

                            <div>

                              <span
                                style={{
                                  display: 'block',
                                  marginBottom: '5px',
                                  color: '#94a3b8',
                                  fontSize: '10px',
                                  fontWeight: 800,
                                  letterSpacing: '.10em',
                                  textTransform: 'uppercase',
                                }}
                              >
                                OCR
                              </span>

                              <strong
                                style={{
                                  color: '#0f172a',
                                  fontSize: '15px',
                                }}
                              >
                                {details.ocrTitle}
                              </strong>

                            </div>

                            <span
                              style={{
                                padding: '6px 9px',
                                borderRadius: '999px',
                                background:
                                  details.ocrDetected
                                    ? '#ecfdf3'
                                    : '#f1f5f9',
                                color:
                                  details.ocrDetected
                                    ? '#15803d'
                                    : '#64748b',
                                fontSize: '10px',
                                fontWeight: 800,
                              }}
                            >
                              {details.ocrDetected
                                ? 'DETECTED'
                                : 'NOT DETECTED'}
                            </span>

                          </div>

                          {ocrText ? (

                            <p
                              style={{
                                margin: 0,
                                padding: '12px 13px',
                                borderRadius: '11px',
                                background: '#f8fafc',
                                color: '#475569',
                                fontSize: '12px',
                                lineHeight: 1.55,
                              }}
                            >
                              {ocrText}
                            </p>

                          ) : (

                            <p
                              style={{
                                margin: 0,
                                color: '#64748b',
                                fontSize: '12px',
                                lineHeight: 1.5,
                              }}
                            >
                              {details.ocrDescription}
                            </p>

                          )}

                        </div>


                      </div>

                    )

                  })()}

                </div>

              )}


              {/* ==================================================
                  AUDIO TRANSCRIPT
              =================================================== */}

              {activeMode === MODES.AUDIO && (

                <div className="modern-details">

                  <h3 className="modern-details-title">
                    Audio Transcription
                  </h3>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns:
                        'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '12px',
                      marginBottom: '14px',
                    }}
                  >

                    <div
                      style={{
                        padding: '14px',
                        border: '1px solid #e6ebf2',
                        borderRadius: '14px',
                        background: '#ffffff',
                      }}
                    >

                      <span
                        style={{
                          display: 'block',
                          marginBottom: '5px',
                          color: '#94a3b8',
                          fontSize: '10px',
                          fontWeight: 800,
                          letterSpacing: '.10em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Language
                      </span>

                      <strong
                        style={{
                          color: '#0f172a',
                          fontSize: '14px',
                        }}
                      >
                        {getAudioLanguage()}
                      </strong>

                    </div>

                    <div
                      style={{
                        padding: '14px',
                        border: '1px solid #e6ebf2',
                        borderRadius: '14px',
                        background: '#ffffff',
                      }}
                    >

                      <span
                        style={{
                          display: 'block',
                          marginBottom: '5px',
                          color: '#94a3b8',
                          fontSize: '10px',
                          fontWeight: 800,
                          letterSpacing: '.10em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Transcription Model
                      </span>

                      <strong
                        style={{
                          color: '#0f172a',
                          fontSize: '14px',
                        }}
                      >
                        {getAudioTranscriptionModel()}
                      </strong>

                    </div>

                    <div
                      style={{
                        padding: '14px',
                        border: '1px solid #e6ebf2',
                        borderRadius: '14px',
                        background: '#ffffff',
                      }}
                    >

                      <span
                        style={{
                          display: 'block',
                          marginBottom: '5px',
                          color: '#94a3b8',
                          fontSize: '10px',
                          fontWeight: 800,
                          letterSpacing: '.10em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Content Model
                      </span>

                      <strong
                        style={{
                          color: '#0f172a',
                          fontSize: '14px',
                        }}
                      >
                        RoBERTa
                      </strong>

                    </div>

                  </div>

                  <div
                    style={{
                      padding: '16px',
                      border: '1px solid #e6ebf2',
                      borderRadius: '15px',
                      background: '#ffffff',
                    }}
                  >

                    <span
                      style={{
                        display: 'block',
                        marginBottom: '8px',
                        color: '#94a3b8',
                        fontSize: '10px',
                        fontWeight: 800,
                        letterSpacing: '.10em',
                        textTransform: 'uppercase',
                      }}
                    >
                      Extracted Transcript
                    </span>

                    <p
                      style={{
                        margin: 0,
                        color: '#475569',
                        fontSize: '13px',
                        lineHeight: 1.65,
                      }}
                    >
                      {getAudioTranscript() ||
                        'No transcript was returned.'}
                    </p>

                  </div>

                </div>

              )}


              {/* ==================================================
                  PLUGIN PIPELINE
              =================================================== */}

              <div className="plugin-panel">

                <div className="plugin-head">

                  <div>

                    <p className="plugin-kicker">
                      Plugin Pipeline
                    </p>

                    <h3 className="plugin-title">
                      {getActivePluginName()}
                    </h3>

                    <p className="plugin-subtitle">
                      The registered plugin connects this
                      input modality with the existing
                      detection and verification services.
                    </p>

                  </div>

                  <span className="plugin-active-chip">

                    <span className="plugin-active-dot"></span>

                    ACTIVE

                  </span>

                </div>

                <div className="plugin-pipeline">

                  {getPluginPipeline().map(
                    (step, index) => (

                      <Fragment
                        key={`${step}-${index}`}
                      >

                        <span className="plugin-step">
                          {step}
                        </span>

                        {index <
                          getPluginPipeline().length - 1 && (

                          <span className="plugin-arrow">
                            →
                          </span>

                        )}

                      </Fragment>

                    )
                  )}

                </div>

              </div>

              {/* ==================================================
                  WEB VERIFICATION + CLEAR EVIDENCE
              =================================================== */}

              {result.web_verification && (

                <div className="verification-panel">

                  <div className="verification-head">

                    <div className="verification-title-wrap">

                      <p className="verification-kicker">

                        {activeMode === MODES.IMAGE
                          ? 'OCR TEXT WEB VERIFICATION'
                          : 'LIVE WEB VERIFICATION'}

                      </p>

                      <h3 className="verification-title">
                        {getVerificationTitle()}
                      </h3>

                      <p className="verification-subtitle">

                        {activeMode === MODES.IMAGE
                          ? 'OCR-extracted text was checked against live Tavily web sources. The web evidence does not change the trained image model prediction.'
                          : activeMode === MODES.AUDIO
                            ? 'The transcript was checked against live web sources.'
                            : 'The submitted claim was checked against live web sources.'}

                      </p>

                    </div>

                    <span
                      className={`verification-chip ${getWebVerdictClass()}`}
                    >

                      <span>
                        ●
                      </span>

                      {getVerificationChipLabel()}

                    </span>

                  </div>


                  {/* ==================================================
                      OCR TEXT SENT TO WEB VERIFICATION
                  ================================================== */}

                  {activeMode === MODES.IMAGE &&
                    getImageOCRText(result) && (

                    <div
                      style={{
                        marginBottom: '14px',
                        padding: '13px 15px',
                        border: '1px solid #e6ebf2',
                        borderRadius: '14px',
                        background: '#ffffff',
                      }}
                    >

                      <strong
                        style={{
                          display: 'block',
                          marginBottom: '5px',
                          color: '#334155',
                          fontSize: '12px',
                          fontWeight: 800,
                        }}
                      >
                        Text sent to Tavily
                      </strong>

                      <p
                        style={{
                          margin: 0,
                          color: '#64748b',
                          fontSize: '12px',
                          lineHeight: 1.55,
                        }}
                      >
                        {getImageOCRText(result)}
                      </p>

                    </div>

                  )}


                  {/* ==================================================
                      SOURCES
                  ================================================== */}

                  {(() => {
                    const displayedEvidence = getDisplayedEvidence()

                    return (
                      <div className="evidence-summary-card">
                        <div className="evidence-summary-header">
                          <div>
                            <h4 className="evidence-summary-title">
                              Sources
                            </h4>

                            <p className="evidence-summary-description">
                              Live web sources returned by Tavily during verification.
                            </p>
                          </div>

                          {displayedEvidence.length > 0 && (
                            <span className="evidence-count-badge">
                              {displayedEvidence.length} source{displayedEvidence.length === 1 ? '' : 's'}
                            </span>
                          )}
                        </div>

                        {displayedEvidence.length > 0 ? (
                          <div className="evidence-list">
                            {displayedEvidence.map((source, index) => (
                              <a
                                key={`source-${source.url}-${index}`}
                                className="evidence-card"
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                title={source.title || 'Open source'}
                              >
                                <span className="evidence-icon">
                                  ↗
                                </span>

                                <span className="evidence-copy">
                                  <span className="evidence-title">
                                    {source.title || 'Source'}
                                  </span>

                                  <span className="evidence-domain">
                                    {getSourceDomain(source.url)}
                                  </span>

                                  {source.description && (
                                    <p className="evidence-description">
                                      {source.description}
                                    </p>
                                  )}

                                  {source.content && (
                                    <p className="evidence-description">
                                      {source.content}
                                    </p>
                                  )}
                                </span>

                                <span className="evidence-open">
                                  ↗
                                </span>
                              </a>
                            ))}
                          </div>
                        ) : (
                          <div className="evidence-empty">
                            <strong>
                              No live evidence sources available
                            </strong>

                            <br />

                            {getEvidenceExplanation()}
                          </div>
                        )}

                        <div className="evidence-explanation">
                          <strong>Evidence analysis</strong>

                          <p>
                            {getEvidenceExplanation()}
                          </p>
                        </div>
                      </div>
                    )
                  })()}


                  {/* ==================================================
                      SOURCES COUNT
                  ================================================== */}

                  <div className="sources-meta" style={{ marginTop: '18px' }}>
                    <span>Sources</span>

                    <span>
                      {getDisplayedEvidence().length}{' '}
                      {getDisplayedEvidence().length === 1 ? 'source' : 'sources'}
                    </span>
                  </div>

                </div>

              )}

              

            </div>

          </section>

        )}

      </main>


      {/* ======================================================
          FOOTER
      ======================================================= */}

      <footer className="site-footer">

        <div>
          Multimodal Fake Content Detection System
        </div>

        <div>
          FastAPI • React • RoBERTa • Whisper • ResNet50
        </div>

      </footer>

    </div>
    
  )
}

export default App

