// ============================================================
// API CONFIGURATION
// ============================================================

const API_BASE_URL =
  'http://127.0.0.1:8000';


// ============================================================
// DOM ELEMENT REFERENCES
// ============================================================

const elements = {

  // HEADER

  pageTitle:
    document.getElementById('pageTitle'),

  pageUrl:
    document.getElementById('pageUrl'),

  backendStatus:
    document.getElementById('backendStatus'),


  // MODE BUTTONS

  textModeButton:
    document.getElementById('textModeButton'),

  imageModeButton:
    document.getElementById('imageModeButton'),

  audioModeButton:
    document.getElementById('audioModeButton'),


  // MODE SECTIONS

  textMode:
    document.getElementById('textMode'),

  imageMode:
    document.getElementById('imageMode'),

  audioMode:
    document.getElementById('audioMode'),


  // TEXT

  analyzeSelectionButton:
    document.getElementById('analyzeSelectionButton'),

  analyzePageButton:
    document.getElementById('analyzePageButton'),

  textPreview:
    document.getElementById('textPreview'),

  textContent:
    document.getElementById('textContent'),


  // IMAGE

  imageUploadZone:
    document.getElementById('imageUploadZone'),

  imageFileInput:
    document.getElementById('imageFileInput'),

  imagePreview:
    document.getElementById('imagePreview'),

  selectedImagePreview:
    document.getElementById('selectedImagePreview'),

  selectedImageName:
    document.getElementById('selectedImageName'),

  selectedImageSize:
    document.getElementById('selectedImageSize'),

  removeImageButton:
    document.getElementById('removeImageButton'),

  analyzeImageButton:
    document.getElementById('analyzeImageButton'),

  pageImages:
    document.getElementById('pageImages'),


  // AUDIO

  audioUploadZone:
    document.getElementById('audioUploadZone'),

  audioFileInput:
    document.getElementById('audioFileInput'),

  audioPreview:
    document.getElementById('audioPreview'),

  selectedAudioPreview:
    document.getElementById('selectedAudioPreview'),

  selectedAudioName:
    document.getElementById('selectedAudioName'),

  selectedAudioSize:
    document.getElementById('selectedAudioSize'),

  removeAudioButton:
    document.getElementById('removeAudioButton'),

  analyzeAudioButton:
    document.getElementById('analyzeAudioButton'),


  // LOADING

  loading:
    document.getElementById('loading'),

  loadingText:
    document.getElementById('loadingText'),


  // ERROR

  error:
    document.getElementById('error'),

  errorText:
    document.getElementById('errorText'),


  // RESULT

  result:
    document.getElementById('result'),

  predictionBox:
    document.getElementById('predictionBox'),

  prediction:
    document.getElementById('prediction'),

  confidence:
    document.getElementById('confidence'),

  modality:
    document.getElementById('modality'),

  model:
    document.getElementById('model'),

  probabilities:
    document.getElementById('probabilities'),

  probabilityList:
    document.getElementById('probabilityList'),


  // SOURCES

  sourcesSection:
    document.getElementById('sourcesSection'),

  sourcesList:
    document.getElementById('sourcesList'),


  // IMAGE DETAILS

  imageDetails:
    document.getElementById('imageDetails'),

  analysisCase:
    document.getElementById('analysisCase'),

  fusionMethodRow:
    document.getElementById('fusionMethodRow'),

  fusionMethod:
    document.getElementById('fusionMethod'),

  ocrTextRow:
    document.getElementById('ocrTextRow'),

  ocrText:
    document.getElementById('ocrText'),

  textModelRow:
    document.getElementById('textModelRow'),

  textModelPrediction:
    document.getElementById('textModelPrediction'),

  textModelConfidence:
    document.getElementById('textModelConfidence'),

  visualModelRow:
    document.getElementById('visualModelRow'),

  visualModelPrediction:
    document.getElementById('visualModelPrediction'),

  visualModelConfidence:
    document.getElementById('visualModelConfidence'),


  // AUDIO DETAILS

  audioDetails:
    document.getElementById('audioDetails'),

  transcriptionModelRow:
    document.getElementById('transcriptionModelRow'),

  transcriptionModel:
    document.getElementById('transcriptionModel'),

  languageRow:
    document.getElementById('languageRow'),

  audioLanguage:
    document.getElementById('audioLanguage'),

  transcriptRow:
    document.getElementById('transcriptRow'),

  audioTranscript:
    document.getElementById('audioTranscript')

};


// ============================================================
// APPLICATION STATE
// ============================================================

let selectedImageFile = null;

let selectedImageObjectUrl = '';

let selectedAudioFile = null;

let selectedAudioObjectUrl = '';

let activeTabId = null;


window.pageInfo = {

  title: '',

  url: '',

  selectedText: '',

  pageText: '',

  images: []

};


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener(
  'DOMContentLoaded',
  async () => {

    setupEvents();

    await loadPageInfo();

    await loadStoredSelectedText();

    await checkBackend();

  }
);


// ============================================================
// EVENT SETUP
// ============================================================

function setupEvents() {


  // TEXT MODE

  elements.textModeButton?.addEventListener(
    'click',
    () => {

      switchMode('text');

    }
  );


  // IMAGE MODE

  elements.imageModeButton?.addEventListener(
    'click',
    () => {

      switchMode('image');

    }
  );


  // AUDIO MODE

  elements.audioModeButton?.addEventListener(
    'click',
    () => {

      switchMode('audio');

    }
  );


  // ANALYZE SELECTED TEXT

  elements.analyzeSelectionButton?.addEventListener(
    'click',
    async () => {

      await loadPageInfo();


      const storedSelection =
        await getStoredSelectedText();


      const selectedText =

        window.pageInfo?.selectedText?.trim() ||

        storedSelection ||

        '';


      if (!selectedText) {

        showError(
          'Select some text on the webpage first.'
        );

        return;

      }


      await analyzeText(
        selectedText
      );

    }
  );


  // ANALYZE CURRENT PAGE

  elements.analyzePageButton?.addEventListener(
    'click',
    async () => {

      await loadPageInfo();


      const pageText =
        window.pageInfo?.pageText?.trim() ||
        '';


      if (!pageText) {

        showError(
          'No readable text was found on this page.'
        );

        return;

      }


      await analyzeText(
        pageText
      );

    }
  );


  // IMAGE UPLOAD

  elements.imageUploadZone?.addEventListener(
    'click',
    () => {

      elements.imageFileInput?.click();

    }
  );


  elements.imageFileInput?.addEventListener(
    'change',
    (event) => {

      const file =
        event.target.files?.[0];


      handleImageFile(
        file
      );

    }
  );


  elements.removeImageButton?.addEventListener(
    'click',
    (event) => {

      event.stopPropagation();

      clearImage();

    }
  );


  elements.analyzeImageButton?.addEventListener(
    'click',
    async () => {

      await analyzeImage();

    }
  );


  // AUDIO UPLOAD

  elements.audioUploadZone?.addEventListener(
    'click',
    () => {

      elements.audioFileInput?.click();

    }
  );


  elements.audioFileInput?.addEventListener(
    'change',
    (event) => {

      const file =
        event.target.files?.[0];


      handleAudioFile(
        file
      );

    }
  );


  elements.removeAudioButton?.addEventListener(
    'click',
    (event) => {

      event.stopPropagation();

      clearAudio();

    }
  );


  elements.analyzeAudioButton?.addEventListener(
    'click',
    async () => {

      await analyzeAudio();

    }
  );

}


// ============================================================
// MODE SWITCHING
// ============================================================

function switchMode(mode) {

  hideError();

  hideResult();


  const buttons = [

    elements.textModeButton,

    elements.imageModeButton,

    elements.audioModeButton

  ];


  buttons.forEach(
    (button) => {

      if (!button) {

        return;

      }


      button.classList.remove(
        'active'
      );


      button.setAttribute(
        'aria-selected',
        'false'
      );

    }
  );


  const sections = [

    elements.textMode,

    elements.imageMode,

    elements.audioMode

  ];


  sections.forEach(
    (section) => {

      section?.classList.add(
        'hidden'
      );

    }
  );


  if (mode === 'text') {

    elements.textModeButton?.classList.add(
      'active'
    );

    elements.textModeButton?.setAttribute(
      'aria-selected',
      'true'
    );

    elements.textMode?.classList.remove(
      'hidden'
    );

  }


  else if (mode === 'image') {

    elements.imageModeButton?.classList.add(
      'active'
    );

    elements.imageModeButton?.setAttribute(
      'aria-selected',
      'true'
    );

    elements.imageMode?.classList.remove(
      'hidden'
    );

    renderPageImages();

  }


  else if (mode === 'audio') {

    elements.audioModeButton?.classList.add(
      'active'
    );

    elements.audioModeButton?.setAttribute(
      'aria-selected',
      'true'
    );

    elements.audioMode?.classList.remove(
      'hidden'
    );

  }

}


// ============================================================
// GET ACTIVE TAB
// ============================================================

async function getActiveTab() {

  const tabs =
    await chrome.tabs.query({
      active: true,
      currentWindow: true
    });


  return tabs[0] || null;

}


// ============================================================
// CHECK WHETHER TAB SUPPORTS CONTENT SCRIPTS
// ============================================================

function isSupportedPage(url) {

  if (!url) {

    return false;

  }


  return (

    url.startsWith('http://') ||

    url.startsWith('https://') ||

    url.startsWith('file://')

  );

}


// ============================================================
// SEND MESSAGE TO CONTENT SCRIPT
// ============================================================

async function sendMessageToContentScript(
  tabId,
  message
) {

  return await chrome.tabs.sendMessage(
    tabId,
    message
  );

}


// ============================================================
// INJECT CONTENT SCRIPT
// ============================================================

async function injectContentScript(tabId) {

  try {

    await chrome.scripting.executeScript({

      target: {
        tabId
      },

      files: [
        'content.js'
      ]

    });


    return true;

  }

  catch {

    return false;

  }

}


// ============================================================
// GET PAGE INFO FROM CONTENT SCRIPT
// AUTOMATICALLY INJECTS CONTENT SCRIPT IF NEEDED
// ============================================================

async function getPageInfoFromContentScript(tab) {

  if (
    !tab?.id ||
    !isSupportedPage(tab.url)
  ) {

    return null;

  }


  try {

    return await sendMessageToContentScript(
      tab.id,
      {
        type:
          'GET_PAGE_INFO'
      }
    );

  }

  catch {


    // Content script is not currently available.
    // Try injecting it dynamically.

    const injected =
      await injectContentScript(
        tab.id
      );


    if (!injected) {

      return null;

    }


    try {

      return await sendMessageToContentScript(
        tab.id,
        {
          type:
            'GET_PAGE_INFO'
        }
      );

    }

    catch {

      return null;

    }

  }

}


// ============================================================
// LOAD PAGE INFORMATION
// ============================================================

async function loadPageInfo() {

  try {

    const tab =
      await getActiveTab();


    if (!tab?.id) {

      return;

    }


    activeTabId =
      tab.id;


    // Basic information available from Chrome

    if (elements.pageTitle) {

      elements.pageTitle.textContent =
        tab.title ||
        'Current page';

    }


    if (elements.pageUrl) {

      elements.pageUrl.textContent =
        tab.url ||
        '—';

    }


    // Reset only page-specific data

    window.pageInfo = {

      title:
        tab.title ||
        '',

      url:
        tab.url ||
        '',

      selectedText:
        '',

      pageText:
        '',

      images:
        []

    };


    // Chrome restricted pages

    if (!isSupportedPage(tab.url)) {

      updateSelectionPreview();

      renderPageImages();

      return;

    }


    const response =
      await getPageInfoFromContentScript(
        tab
      );


    if (!response) {

      updateSelectionPreview();

      renderPageImages();

      return;

    }


    window.pageInfo = {

      ...window.pageInfo,

      ...response

    };


    if (elements.pageTitle) {

      elements.pageTitle.textContent =

        response.title ||

        tab.title ||

        'Current page';

    }


    if (elements.pageUrl) {

      elements.pageUrl.textContent =

        response.url ||

        tab.url ||

        '—';

    }


    updateSelectionPreview();

    renderPageImages();

  }

  catch {

    // Graceful fallback.
    // Do not log this as an extension error.

    updateSelectionPreview();

    renderPageImages();

  }

}


// ============================================================
// GET STORED SELECTED TEXT
// ============================================================

async function getStoredSelectedText() {

  try {

    const stored =
      await chrome.storage.local.get(
        'selectedText'
      );


    return (
      stored.selectedText ||
      ''
    ).trim();

  }

  catch {

    return '';

  }

}


// ============================================================
// LOAD STORED SELECTED TEXT
// ============================================================

async function loadStoredSelectedText() {

  const storedText =
    await getStoredSelectedText();


  if (!storedText) {

    return;

  }


  if (
    !window.pageInfo.selectedText?.trim()
  ) {

    window.pageInfo.selectedText =
      storedText;


    updateSelectionPreview();

  }

}


// ============================================================
// UPDATE TEXT SELECTION PREVIEW
// ============================================================

function updateSelectionPreview() {

  if (
    !elements.textPreview ||
    !elements.textContent
  ) {

    return;

  }


  const selectedText =
    window.pageInfo.selectedText?.trim() ||
    '';


  if (selectedText) {

    elements.textPreview.classList.remove(
      'hidden'
    );


    elements.textContent.textContent =
      selectedText;

  }

  else {

    elements.textPreview.classList.add(
      'hidden'
    );

  }

}


// ============================================================
// RENDER PAGE IMAGES
// ============================================================

function renderPageImages() {

  if (!elements.pageImages) {

    return;

  }


  const images =
    Array.isArray(
      window.pageInfo?.images
    )
      ? window.pageInfo.images
      : [];


  elements.pageImages.innerHTML =
    '';


  if (!images.length) {

    const empty =
      document.createElement(
        'div'
      );


    empty.className =
      'no-images';


    empty.textContent =
      'No suitable images found on this page.';


    elements.pageImages.appendChild(
      empty
    );


    return;

  }


  images.forEach(
    (image, index) => {

      const imageCard =
        document.createElement(
          'div'
        );


      imageCard.className =
        'page-image-card';


      const preview =
        document.createElement(
          'img'
        );


      preview.src =
        image.src;


      preview.alt =

        image.alt ||

        image.name ||

        `Page image ${index + 1}`;


      preview.addEventListener(
        'error',
        () => {

          imageCard.remove();

        }
      );


      const imageInfo =
        document.createElement(
          'div'
        );


      imageInfo.className =
        'page-image-info';


      const imageName =
        document.createElement(
          'div'
        );


      imageName.className =
        'page-image-name';


      imageName.textContent =

        image.name ||

        image.alt ||

        `Image ${index + 1}`;


      const imageDimensions =
        document.createElement(
          'div'
        );


      imageDimensions.className =
        'page-image-dimensions';


      imageDimensions.textContent =
        `${image.width || '?'} × ${image.height || '?'}`;


      const analyzeButton =
        document.createElement(
          'button'
        );


      analyzeButton.type =
        'button';


      analyzeButton.className =
        'secondary-button';


      analyzeButton.textContent =
        'Analyze This Image';


      analyzeButton.addEventListener(
        'click',
        async () => {

          await analyzePageImage(
            image.src
          );

        }
      );


      imageInfo.appendChild(
        imageName
      );


      imageInfo.appendChild(
        imageDimensions
      );


      imageCard.appendChild(
        preview
      );


      imageCard.appendChild(
        imageInfo
      );


      imageCard.appendChild(
        analyzeButton
      );


      elements.pageImages.appendChild(
        imageCard
      );

    }
  );

}


// ============================================================
// BACKEND HEALTH CHECK
// ============================================================

async function checkBackend() {

  try {

    const response =
      await fetch(
        `${API_BASE_URL}/health`
      );


    if (!response.ok) {

      throw new Error(
        'Backend unavailable'
      );

    }


    if (elements.backendStatus) {

      elements.backendStatus.textContent =
        'Backend Connected';


      elements.backendStatus.classList.remove(
        'offline'
      );


      elements.backendStatus.classList.add(
        'online'
      );

    }


    return true;

  }

  catch {

    if (elements.backendStatus) {

      elements.backendStatus.textContent =
        'Backend Offline';


      elements.backendStatus.classList.remove(
        'online'
      );


      elements.backendStatus.classList.add(
        'offline'
      );

    }


    return false;

  }

}


// ============================================================
// TEXT ANALYSIS
// ============================================================

async function analyzeText(text) {

  const cleanText =
    String(text || '')
      .trim();


  if (!cleanText) {

    showError(
      'Please provide some text to analyze.'
    );

    return;

  }


  hideError();

  hideResult();

  showLoading(
    'Analyzing text and verifying evidence...'
  );


  try {

    const formData =
      new FormData();


    formData.append(
      'text',
      cleanText
    );


    const response =
      await fetch(
        `${API_BASE_URL}/api/v1/verify/text`,
        {
          method: 'POST',
          body: formData
        }
      );


    const data =
      await parseResponse(
        response
      );


    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          data,
          'Text analysis failed.'
        )
      );

    }


    data.modality =
      data.modality ||
      'text';


    displayResult(
      data
    );

  }

  catch (error) {

    showError(
      error.message ||
      'Unable to analyze text.'
    );

  }

  finally {

    hideLoading();

  }

}


// ============================================================
// HANDLE IMAGE FILE
// ============================================================

function handleImageFile(file) {

  if (!file) {

    return;

  }


  if (
    !file.type.startsWith(
      'image/'
    )
  ) {

    showError(
      'Please select a valid image file.'
    );

    return;

  }


  clearImage();


  selectedImageFile =
    file;


  selectedImageObjectUrl =
    URL.createObjectURL(
      file
    );


  if (elements.selectedImagePreview) {

    elements.selectedImagePreview.src =
      selectedImageObjectUrl;

  }


  if (elements.selectedImageName) {

    elements.selectedImageName.textContent =
      file.name;

  }


  if (elements.selectedImageSize) {

    elements.selectedImageSize.textContent =
      formatFileSize(
        file.size
      );

  }


  elements.imagePreview?.classList.remove(
    'hidden'
  );

}


// ============================================================
// CLEAR IMAGE
// ============================================================

function clearImage() {

  selectedImageFile =
    null;


  if (selectedImageObjectUrl) {

    URL.revokeObjectURL(
      selectedImageObjectUrl
    );


    selectedImageObjectUrl =
      '';

  }


  if (elements.imageFileInput) {

    elements.imageFileInput.value =
      '';

  }


  if (elements.selectedImagePreview) {

    elements.selectedImagePreview.src =
      '';

  }


  if (elements.selectedImageName) {

    elements.selectedImageName.textContent =
      '';

  }


  if (elements.selectedImageSize) {

    elements.selectedImageSize.textContent =
      '';

  }


  elements.imagePreview?.classList.add(
    'hidden'
  );

}


// ============================================================
// ANALYZE UPLOADED IMAGE
// ============================================================

async function analyzeImage() {

  if (!selectedImageFile) {

    showError(
      'Please select an image first.'
    );

    return;

  }


  hideError();

  hideResult();

  showLoading(
    'Analyzing image using the trained AI model...'
  );


  try {

    const formData =
      new FormData();


    formData.append(
      'file',
      selectedImageFile
    );


    const response =
      await fetch(
        `${API_BASE_URL}/api/v1/predict/image`,
        {
          method: 'POST',
          body: formData
        }
      );


    const data =
      await parseResponse(
        response
      );


    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          data,
          'Image analysis failed.'
        )
      );

    }


    data.modality =
      data.modality ||
      'image';


    displayResult(
      data
    );

  }

  catch (error) {

    showError(
      error.message ||
      'Unable to analyze image.'
    );

  }

  finally {

    hideLoading();

  }

}


// ============================================================
// ANALYZE IMAGE FROM WEBPAGE
// ============================================================

async function analyzePageImage(imageUrl) {

  if (!imageUrl) {

    showError(
      'Invalid image URL.'
    );

    return;

  }


  hideError();

  hideResult();

  showLoading(
    'Downloading and analyzing webpage image...'
  );


  try {

    const imageResponse =
      await fetch(
        imageUrl
      );


    if (!imageResponse.ok) {

      throw new Error(
        'Unable to download the selected webpage image.'
      );

    }


    const imageBlob =
      await imageResponse.blob();


    const filename =
      getFilenameFromUrl(
        imageUrl
      );


    const imageFile =
      new File(
        [imageBlob],
        filename,
        {
          type:
            imageBlob.type ||
            'image/jpeg'
        }
      );


    const formData =
      new FormData();


    formData.append(
      'file',
      imageFile
    );


    const response =
      await fetch(
        `${API_BASE_URL}/api/v1/predict/image`,
        {
          method: 'POST',
          body: formData
        }
      );


    const data =
      await parseResponse(
        response
      );


    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          data,
          'Webpage image analysis failed.'
        )
      );

    }


    data.modality =
      data.modality ||
      'image';


    displayResult(
      data
    );

  }

  catch (error) {

    showError(
      error.message ||
      'Unable to analyze webpage image.'
    );

  }

  finally {

    hideLoading();

  }

}


// ============================================================
// HANDLE AUDIO FILE
// ============================================================

function handleAudioFile(file) {

  if (!file) {

    return;

  }


  if (
    !file.type.startsWith(
      'audio/'
    )
  ) {

    showError(
      'Please select a valid audio file.'
    );

    return;

  }


  clearAudio();


  selectedAudioFile =
    file;


  selectedAudioObjectUrl =
    URL.createObjectURL(
      file
    );


  if (elements.selectedAudioPreview) {

    elements.selectedAudioPreview.src =
      selectedAudioObjectUrl;

  }


  if (elements.selectedAudioName) {

    elements.selectedAudioName.textContent =
      file.name;

  }


  if (elements.selectedAudioSize) {

    elements.selectedAudioSize.textContent =
      formatFileSize(
        file.size
      );

  }


  elements.audioPreview?.classList.remove(
    'hidden'
  );

}


// ============================================================
// CLEAR AUDIO
// ============================================================

function clearAudio() {

  selectedAudioFile =
    null;


  if (selectedAudioObjectUrl) {

    URL.revokeObjectURL(
      selectedAudioObjectUrl
    );


    selectedAudioObjectUrl =
      '';

  }


  if (elements.audioFileInput) {

    elements.audioFileInput.value =
      '';

  }


  if (elements.selectedAudioPreview) {

    elements.selectedAudioPreview.src =
      '';

  }


  if (elements.selectedAudioName) {

    elements.selectedAudioName.textContent =
      '';

  }


  if (elements.selectedAudioSize) {

    elements.selectedAudioSize.textContent =
      '';

  }


  elements.audioPreview?.classList.add(
    'hidden'
  );

}


// ============================================================
// ANALYZE AUDIO
// ============================================================

async function analyzeAudio() {

  if (!selectedAudioFile) {

    showError(
      'Please select an audio file first.'
    );

    return;

  }


  hideError();

  hideResult();

  showLoading(
    'Transcribing and verifying audio content...'
  );


  try {

    const formData =
      new FormData();


    formData.append(
      'file',
      selectedAudioFile
    );


    const response =
      await fetch(
        `${API_BASE_URL}/api/v1/verify/audio`,
        {
          method: 'POST',
          body: formData
        }
      );


    const data =
      await parseResponse(
        response
      );


    if (!response.ok) {

      throw new Error(
        getErrorMessage(
          data,
          'Audio analysis failed.'
        )
      );

    }


    data.modality =
      data.modality ||
      'audio';


    displayResult(
      data
    );

  }

  catch (error) {

    showError(
      error.message ||
      'Unable to analyze audio.'
    );

  }

  finally {

    hideLoading();

  }

}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

  if (!elements.result) {

    return;

  }


  const resultData =

    (
      data?.result &&
      typeof data.result === 'object'
    )

      ? {

          ...data,

          ...data.result,

          modality:

            data.modality ||

            data.result.modality

        }

      : data;


  elements.result.classList.remove(
    'hidden'
  );


  const prediction =
    getFinalPrediction(
      resultData
    );


  const confidence =
    getDisplayConfidence(
      resultData
    );


  if (elements.predictionBox) {

    elements.predictionBox.classList.remove(
      'real',
      'fake',
      'unknown'
    );


    const normalizedPrediction =
      String(prediction)
        .toUpperCase();


    if (

      normalizedPrediction.includes('FAKE') ||

      normalizedPrediction.includes('FALSE') ||

      normalizedPrediction.includes('MISINFORMATION') ||

      normalizedPrediction.includes('AI-GENERATED')

    ) {

      elements.predictionBox.classList.add(
        'fake'
      );

    }


    else if (

      normalizedPrediction.includes('REAL') ||

      normalizedPrediction.includes('TRUE') ||

      normalizedPrediction.includes('GENUINE')

    ) {

      elements.predictionBox.classList.add(
        'real'
      );

    }

  }


  if (elements.prediction) {

    elements.prediction.textContent =
      prediction;

  }


  if (elements.confidence) {

    elements.confidence.textContent =

      Number.isNaN(confidence)

        ? 'Confidence unavailable'

        : `${confidence.toFixed(2)}% confidence`;

  }


  if (elements.modality) {

    elements.modality.textContent =
      formatModality(
        resultData.modality ||
        'unknown'
      );

  }


  if (elements.model) {

    elements.model.textContent =
      getDisplayModel(
        resultData
      );

  }


  hideImageDetails();

  hideAudioDetails();


  if (
    String(
      resultData.modality
    ).toLowerCase() === 'image'
  ) {

    displayImageDetails(
      resultData
    );

  }


  if (
    String(
      resultData.modality
    ).toLowerCase() === 'audio'
  ) {

    displayAudioDetails(
      resultData
    );

  }


  renderProbabilities(
    getDisplayProbabilities(
      resultData
    )
  );


  renderSources(
    getDisplaySources(
      resultData
    )
  );

}


// ============================================================
// GET FINAL PREDICTION
// ============================================================

function getFinalPrediction(data) {

  const value =

    data?.final_prediction ||

    data?.prediction ||

    data?.label ||

    data?.verdict ||

    data?.short_prediction ||

    data?.classification?.prediction ||

    data?.classification?.label ||

    data?.classification?.verdict ||

    'UNKNOWN';


  return formatPredictionLabel(
    value
  );

}


// ============================================================
// GET CONFIDENCE
// ============================================================

function getDisplayConfidence(data) {

  const candidates = [

    data?.final_confidence_percent,

    data?.confidence_percent,

    data?.confidence,

    data?.confidence_score,

    data?.classification?.confidence_percent,

    data?.classification?.confidence,

    data?.classification?.confidence_score

  ];


  for (const value of candidates) {

    if (

      value === null ||

      value === undefined ||

      value === ''

    ) {

      continue;

    }


    const numericValue =
      Number(value);


    if (
      Number.isFinite(
        numericValue
      )
    ) {

      return normalizeProbability(
        numericValue
      );

    }

  }


  return Number.NaN;

}


// ============================================================
// GET MODEL NAME
// ============================================================

function getDisplayModel(data) {

  if (data?.model) {

    return String(
      data.model
    );

  }


  if (data?.model_name) {

    return String(
      data.model_name
    );

  }


  if (
    data?.classification?.model
  ) {

    return String(
      data.classification.model
    );

  }


  const modality =
    String(
      data?.modality ||
      ''
    ).toLowerCase();


  if (modality === 'text') {

    return 'RoBERTa + Live Web Verification';

  }


  if (modality === 'image') {

    return 'ResNet50 / Multimodal Image Analysis';

  }


  if (modality === 'audio') {

    return 'Whisper + RoBERTa + Live Web Verification';

  }


  return '—';

}


// ============================================================
// GET PROBABILITIES
// ============================================================

function getDisplayProbabilities(data) {

  if (

    data?.probabilities &&

    typeof data.probabilities === 'object' &&

    Object.keys(
      data.probabilities
    ).length

  ) {

    return data.probabilities;

  }


  if (

    data?.classification?.probabilities &&

    typeof data.classification.probabilities === 'object'

  ) {

    return data.classification.probabilities;

  }


  const probabilities = {};


  if (
    data?.fake_probability !== undefined
  ) {

    probabilities.fake =
      data.fake_probability;

  }


  if (
    data?.real_probability !== undefined
  ) {

    probabilities.real =
      data.real_probability;

  }


  return probabilities;

}


// ============================================================
// GET SOURCES
// ============================================================

function getDisplaySources(data) {

  let rawSources =

    data?.sources ||

    data?.source ||

    data?.verification_sources ||

    data?.evidence ||

    data?.references ||

    data?.result?.sources ||

    data?.classification?.sources ||

    [];


  if (!Array.isArray(rawSources)) {

    rawSources =
      [rawSources];

  }


  return rawSources
    .filter(Boolean)
    .slice(0, 2);

}


// ============================================================
// RENDER SOURCES
// ============================================================

function renderSources(sources) {

  if (
    !elements.sourcesSection ||
    !elements.sourcesList
  ) {

    return;

  }


  elements.sourcesList.innerHTML =
    '';


  if (
    !Array.isArray(sources) ||
    sources.length === 0
  ) {

    elements.sourcesSection.classList.add(
      'hidden'
    );

    return;

  }


  elements.sourcesSection.classList.remove(
    'hidden'
  );


  sources.forEach(
    (source, index) => {

      const sourceItem =
        document.createElement(
          'div'
        );


      sourceItem.className =
        'source-item';


      let title =
        `Source ${index + 1}`;


      let url =
        '';


      let snippet =
        '';


      if (
        typeof source === 'string'
      ) {

        url =
          source;


        title =
          getDomainFromUrl(source) ||
          title;

      }


      else if (
        typeof source === 'object'
      ) {

        title =

          source.title ||

          source.name ||

          source.source ||

          source.publisher ||

          title;


        url =

          source.url ||

          source.link ||

          source.href ||

          '';


        snippet =

          source.snippet ||

          source.description ||

          source.text ||

          source.summary ||

          '';

      }


      const titleElement =
        document.createElement(
          'strong'
        );


      titleElement.textContent =
        title;


      sourceItem.appendChild(
        titleElement
      );


      if (snippet) {

        const snippetElement =
          document.createElement(
            'p'
          );


        snippetElement.textContent =
          snippet;


        sourceItem.appendChild(
          snippetElement
        );

      }


      if (url) {

        const linkElement =
          document.createElement(
            'a'
          );


        linkElement.href =
          url;


        linkElement.target =
          '_blank';


        linkElement.rel =
          'noopener noreferrer';


        linkElement.textContent =
          getDomainFromUrl(url) ||
          'Open Source';


        sourceItem.appendChild(
          linkElement
        );

      }


      elements.sourcesList.appendChild(
        sourceItem
      );

    }
  );

}


// ============================================================
// IMAGE DETAILS
// ============================================================

function displayImageDetails(data) {

  if (!elements.imageDetails) {

    return;

  }


  elements.imageDetails.classList.remove(
    'hidden'
  );


  const imageDetails =

    data.image_details ||

    data.details ||

    data;


  const analysisCase =

    imageDetails.analysis_case ||

    data.analysis_case ||

    '—';


  elements.analysisCase.textContent =
    formatAnalysisCase(
      analysisCase
    );


  const fusionMethod =

    imageDetails.fusion_method ||

    data.fusion_method;


  if (fusionMethod) {

    elements.fusionMethodRow?.classList.remove(
      'hidden'
    );

    elements.fusionMethod.textContent =
      fusionMethod;

  }

  else {

    elements.fusionMethodRow?.classList.add(
      'hidden'
    );

  }


  const ocrText =

    imageDetails.ocr_text ||

    data.ocr_text;


  if (ocrText) {

    elements.ocrTextRow?.classList.remove(
      'hidden'
    );

    elements.ocrText.textContent =
      ocrText;

  }

  else {

    elements.ocrTextRow?.classList.add(
      'hidden'
    );

  }


  const textPrediction =

    imageDetails.text_prediction ||

    data.text_prediction;


  const textConfidence =

    imageDetails.text_confidence ??

    data.text_confidence;


  if (textPrediction) {

    elements.textModelRow?.classList.remove(
      'hidden'
    );


    elements.textModelPrediction.textContent =
      formatPredictionLabel(
        textPrediction
      );


    elements.textModelConfidence.textContent =

      textConfidence !== undefined &&
      textConfidence !== null

        ? `${normalizeProbability(
            Number(textConfidence)
          ).toFixed(2)}%`

        : '—';

  }

  else {

    elements.textModelRow?.classList.add(
      'hidden'
    );

  }


  const visualPrediction =

    imageDetails.visual_prediction ||

    data.visual_prediction;


  const visualConfidence =

    imageDetails.visual_confidence ??

    data.visual_confidence;


  if (visualPrediction) {

    elements.visualModelRow?.classList.remove(
      'hidden'
    );


    elements.visualModelPrediction.textContent =
      formatPredictionLabel(
        visualPrediction
      );


    elements.visualModelConfidence.textContent =

      visualConfidence !== undefined &&
      visualConfidence !== null

        ? `${normalizeProbability(
            Number(visualConfidence)
          ).toFixed(2)}%`

        : '—';

  }

  else {

    elements.visualModelRow?.classList.add(
      'hidden'
    );

  }

}


// ============================================================
// AUDIO DETAILS
// ============================================================

function displayAudioDetails(data) {

  if (!elements.audioDetails) {

    return;

  }


  elements.audioDetails.classList.remove(
    'hidden'
  );


  const classification =
    data.classification ||
    {};


  const transcriptionModel =

    data.transcription_model ||

    data.whisper_model ||

    classification.transcription_model ||

    classification.whisper_model;


  const language =

    data.language ||

    classification.language;


  const transcript =

    data.transcript ||

    data.transcription ||

    classification.transcript ||

    classification.transcription ||

    data.verification_text;


  if (transcriptionModel) {

    elements.transcriptionModelRow?.classList.remove(
      'hidden'
    );


    elements.transcriptionModel.textContent =
      transcriptionModel;

  }

  else {

    elements.transcriptionModelRow?.classList.add(
      'hidden'
    );

  }


  if (language) {

    elements.languageRow?.classList.remove(
      'hidden'
    );


    elements.audioLanguage.textContent =
      String(language).toUpperCase();

  }

  else {

    elements.languageRow?.classList.add(
      'hidden'
    );

  }


  if (transcript) {

    elements.transcriptRow?.classList.remove(
      'hidden'
    );


    elements.audioTranscript.textContent =
      transcript;

  }

  else {

    elements.transcriptRow?.classList.add(
      'hidden'
    );

  }

}


// ============================================================
// HIDE IMAGE DETAILS
// ============================================================

function hideImageDetails() {

  elements.imageDetails?.classList.add(
    'hidden'
  );

}


// ============================================================
// HIDE AUDIO DETAILS
// ============================================================

function hideAudioDetails() {

  elements.audioDetails?.classList.add(
    'hidden'
  );

}


// ============================================================
// RENDER PROBABILITIES
// ============================================================

function renderProbabilities(probabilities) {

  if (!elements.probabilityList) {

    return;

  }


  elements.probabilityList.innerHTML =
    '';


  if (

    !probabilities ||

    typeof probabilities !== 'object' ||

    !Object.keys(probabilities).length

  ) {

    const empty =
      document.createElement(
        'div'
      );


    empty.textContent =
      'Probability details unavailable.';


    elements.probabilityList.appendChild(
      empty
    );


    return;

  }


  Object.entries(
    probabilities
  ).forEach(
    ([label, value]) => {

      const probability =
        normalizeProbability(
          Number(value)
        );


      const row =
        document.createElement(
          'div'
        );


      row.className =
        'probability-row';


      const info =
        document.createElement(
          'div'
        );


      info.className =
        'probability-info';


      const labelElement =
        document.createElement(
          'span'
        );


      labelElement.textContent =
        formatPredictionLabel(
          label
        );


      const valueElement =
        document.createElement(
          'strong'
        );


      valueElement.textContent =

        Number.isNaN(probability)

          ? '—'

          : `${probability.toFixed(2)}%`;


      info.appendChild(
        labelElement
      );


      info.appendChild(
        valueElement
      );


      const track =
        document.createElement(
          'div'
        );


      track.className =
        'probability-track';


      const fill =
        document.createElement(
          'div'
        );


      fill.className =
        'probability-fill';


      fill.style.width =

        Number.isNaN(probability)

          ? '0%'

          : `${Math.max(
              0,
              Math.min(
                100,
                probability
              )
            )}%`;


      track.appendChild(
        fill
      );


      row.appendChild(
        info
      );


      row.appendChild(
        track
      );


      elements.probabilityList.appendChild(
        row
      );

    }
  );

}


// ============================================================
// RESPONSE PARSER
// ============================================================

async function parseResponse(response) {

  const contentType =
    response.headers.get(
      'content-type'
    ) ||
    '';


  if (
    contentType.includes(
      'application/json'
    )
  ) {

    return await response.json();

  }


  const text =
    await response.text();


  return {

    detail:
      text

  };

}


// ============================================================
// EXTRACT ERROR MESSAGE
// ============================================================

function getErrorMessage(
  data,
  fallback
) {

  if (
    typeof data === 'string'
  ) {

    return data ||
      fallback;

  }


  if (
    typeof data?.detail === 'string'
  ) {

    return data.detail;

  }


  if (
    Array.isArray(data?.detail)
  ) {

    return data.detail
      .map(
        (item) =>
          item?.msg ||
          JSON.stringify(item)
      )
      .join(', ');

  }


  if (data?.message) {

    return data.message;

  }


  if (data?.error) {

    return data.error;

  }


  return fallback;

}


// ============================================================
// LOADING UI
// ============================================================

function showLoading(message) {

  elements.loading?.classList.remove(
    'hidden'
  );


  if (
    elements.loadingText &&
    message
  ) {

    elements.loadingText.textContent =
      message;

  }

}


function hideLoading() {

  elements.loading?.classList.add(
    'hidden'
  );

}


// ============================================================
// ERROR UI
// ============================================================

function showError(message) {

  if (elements.errorText) {

    elements.errorText.textContent =
      message;

  }


  if (elements.error) {

    elements.error.classList.remove(
      'hidden'
    );

  }

}


function hideError() {

  elements.error?.classList.add(
    'hidden'
  );


  if (elements.errorText) {

    elements.errorText.textContent =
      '';

  }

}


// ============================================================
// HIDE RESULT
// ============================================================

function hideResult() {

  elements.result?.classList.add(
    'hidden'
  );


  if (elements.prediction) {

    elements.prediction.textContent =
      '—';

  }


  if (elements.confidence) {

    elements.confidence.textContent =
      '—';

  }


  if (elements.modality) {

    elements.modality.textContent =
      '—';

  }


  if (elements.model) {

    elements.model.textContent =
      '—';

  }


  elements.predictionBox?.classList.remove(
    'real',
    'fake',
    'unknown'
  );


  hideImageDetails();

  hideAudioDetails();


  if (elements.probabilityList) {

    elements.probabilityList.innerHTML =
      '';

  }


  elements.sourcesSection?.classList.add(
    'hidden'
  );


  if (elements.sourcesList) {

    elements.sourcesList.innerHTML =
      '';

  }

}


// ============================================================
// NORMALIZE PROBABILITY
// ============================================================

function normalizeProbability(value) {

  if (
    !Number.isFinite(value)
  ) {

    return Number.NaN;

  }


  if (
    value >= 0 &&
    value <= 1
  ) {

    return value * 100;

  }


  return value;

}


// ============================================================
// FORMAT FILE SIZE
// ============================================================

function formatFileSize(bytes) {

  if (
    !Number.isFinite(bytes)
  ) {

    return '—';

  }


  if (bytes < 1024) {

    return `${bytes} B`;

  }


  const kb =
    bytes / 1024;


  if (kb < 1024) {

    return `${kb.toFixed(1)} KB`;

  }


  const mb =
    kb / 1024;


  return `${mb.toFixed(2)} MB`;

}


// ============================================================
// FORMAT MODALITY
// ============================================================

function formatModality(modality) {

  const labels = {

    text:
      'Text',

    image:
      'Image',

    audio:
      'Audio'

  };


  const normalized =
    String(
      modality || ''
    ).toLowerCase();


  return (
    labels[normalized] ||
    formatPredictionLabel(modality)
  );

}


// ============================================================
// FORMAT PREDICTION LABEL
// ============================================================

function formatPredictionLabel(label) {

  if (

    label === null ||

    label === undefined ||

    label === ''

  ) {

    return 'UNKNOWN';

  }


  const normalized =
    String(label)
      .trim()
      .toLowerCase();


  // Your project mapping:
  // label_0 = REAL
  // label_1 = FAKE

  if (

    normalized === 'label_0' ||

    normalized === 'label 0' ||

    normalized === '0'

  ) {

    return 'REAL';

  }


  if (

    normalized === 'label_1' ||

    normalized === 'label 1' ||

    normalized === '1'

  ) {

    return 'FAKE';

  }


  return String(label)

    .replace(
      /_/g,
      ' '
    )

    .replace(
      /\b\w/g,
      (character) =>
        character.toUpperCase()
    );

}


// ============================================================
// FORMAT ANALYSIS CASE
// ============================================================

function formatAnalysisCase(value) {

  if (!value) {

    return '—';

  }


  return String(value)

    .replace(
      /_/g,
      ' '
    )

    .replace(
      /\b\w/g,
      (character) =>
        character.toUpperCase()
    );

}


// ============================================================
// GET FILENAME FROM URL
// ============================================================

function getFilenameFromUrl(url) {

  try {

    const parsedUrl =
      new URL(url);


    const filename =
      parsedUrl.pathname
        .split('/')
        .pop();


    if (
      filename &&
      filename.includes('.')
    ) {

      return filename;

    }

  }

  catch {

    // Ignore

  }


  return `webpage-image-${Date.now()}.jpg`;

}