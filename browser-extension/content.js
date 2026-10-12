// ============================================================
// CONTENT SCRIPT
// Multimodal Fake Content Detection Browser Extension
// ============================================================


// ============================================================
// CONSTANTS
// ============================================================

const MAX_PAGE_TEXT_LENGTH = 12000;

const MAX_PAGE_IMAGES = 20;

const MIN_IMAGE_WIDTH = 100;

const MIN_IMAGE_HEIGHT = 100;


// ============================================================
// GET SELECTED TEXT
// ============================================================

function getSelectedText() {

  try {

    const selection = window.getSelection();

    return (
      selection
        ?.toString()
        ?.replace(/\s+/g, ' ')
        .trim() ||
      ''
    );

  }

  catch {

    return '';

  }

}


// ============================================================
// GET PAGE TEXT
// ============================================================

function getPageText() {

  try {

    if (!document.body) {

      return '';

    }


    const clonedBody =
      document.body.cloneNode(true);


    const unwantedElements =
      clonedBody.querySelectorAll(
        [
          'script',
          'style',
          'noscript',
          'svg',
          'canvas',
          'iframe'
        ].join(',')
      );


    unwantedElements.forEach(
      (element) => {

        element.remove();

      }
    );


    const pageText =
      (
        clonedBody.innerText ||
        clonedBody.textContent ||
        ''
      )
        .replace(/\s+/g, ' ')
        .trim();


    return pageText;

  }

  catch {

    return '';

  }

}


// ============================================================
// GET IMAGE SOURCE
// ============================================================

function getImageSource(image) {

  if (!image) {

    return '';

  }


  const possibleSources = [

    image.currentSrc,

    image.src,

    image.getAttribute('src'),

    image.getAttribute('data-src'),

    image.getAttribute('data-lazy-src'),

    image.getAttribute('data-original'),

    image.getAttribute('data-image'),

    image.getAttribute('data-url')

  ];


  const validSource =
    possibleSources.find(
      (source) => {

        return (
          typeof source === 'string' &&
          source.trim() !== ''
        );

      }
    );


  return validSource || '';

}


// ============================================================
// CONVERT URL TO ABSOLUTE URL
// ============================================================

function getAbsoluteUrl(source) {

  try {

    if (!source) {

      return '';

    }


    return new URL(
      source,
      window.location.href
    ).href;

  }

  catch {

    return source || '';

  }

}


// ============================================================
// CHECK VALID IMAGE SOURCE
// ============================================================

function isValidImageSource(source) {

  if (
    !source ||
    typeof source !== 'string'
  ) {

    return false;

  }


  const normalized =
    source
      .trim()
      .toLowerCase();


  if (
    normalized.startsWith('data:')
  ) {

    return false;

  }


  if (
    normalized.startsWith('blob:')
  ) {

    return false;

  }


  return true;

}


// ============================================================
// GET PAGE IMAGES
// ============================================================

function getPageImages() {

  try {

    const images =
      Array.from(document.images);


    const pageImages =
      images
        .map(
          (image, index) => {

            const rawSource =
              getImageSource(image);


            const src =
              getAbsoluteUrl(rawSource);


            const width =
              image.naturalWidth ||
              image.width ||
              image.clientWidth ||
              0;


            const height =
              image.naturalHeight ||
              image.height ||
              image.clientHeight ||
              0;


            let name =
              image.alt ||
              image.title ||
              '';


            if (
              !name &&
              src
            ) {

              try {

                const imageUrl =
                  new URL(src);


                name =
                  imageUrl.pathname
                    .split('/')
                    .pop() ||
                  `Image ${index + 1}`;

              }

              catch {

                name =
                  `Image ${index + 1}`;

              }

            }


            return {

              id:
                `page-image-${index}`,

              src,

              alt:
                image.alt ||
                '',

              title:
                image.title ||
                '',

              name:
                name ||
                `Image ${index + 1}`,

              width,

              height

            };

          }
        )


        // Remove invalid images

        .filter(
          (image) => {

            return (

              isValidImageSource(
                image.src
              ) &&

              image.width >=
              MIN_IMAGE_WIDTH &&

              image.height >=
              MIN_IMAGE_HEIGHT

            );

          }
        )


        // Remove duplicates

        .filter(
          (image, index, array) => {

            return (

              array.findIndex(
                (item) =>
                  item.src ===
                  image.src
              ) === index

            );

          }
        )


        // Limit images

        .slice(
          0,
          MAX_PAGE_IMAGES
        );


    return pageImages;

  }

  catch {

    return [];

  }

}


// ============================================================
// GET COMPLETE PAGE INFORMATION
// ============================================================

function getPageInfo() {

  return {

    success:
      true,


    title:
      document.title ||
      'Untitled page',


    url:
      window.location.href,


    selectedText:
      getSelectedText(),


    pageText:
      getPageText()
        .slice(
          0,
          MAX_PAGE_TEXT_LENGTH
        ),


    images:
      getPageImages(),


    metadata: {

      title:
        document.title ||
        '',

      url:
        window.location.href,

      hostname:
        window.location.hostname

    }

  };

}


// ============================================================
// SAVE SELECTED TEXT
// ============================================================

function saveSelectedTextToStorage(text) {

  try {

    if (!text) {

      return;

    }


    chrome.runtime.sendMessage(
      {

        action:
          'SAVE_SELECTED_TEXT',

        text,

        sourceUrl:
          window.location.href,

        sourceTitle:
          document.title

      },
      () => {

        // Ignore errors silently.
        // Background service worker may temporarily restart.

        void chrome.runtime.lastError;

      }
    );

  }

  catch {

    // Ignore safely

  }

}


// ============================================================
// HANDLE MOUSE TEXT SELECTION
// ============================================================

document.addEventListener(
  'mouseup',
  () => {

    setTimeout(
      () => {

        const selectedText =
          getSelectedText();


        if (
          selectedText &&
          selectedText.length >= 2
        ) {

          saveSelectedTextToStorage(
            selectedText
          );

        }

      },
      50
    );

  }
);


// ============================================================
// HANDLE KEYBOARD TEXT SELECTION
// ============================================================

document.addEventListener(
  'keyup',
  (event) => {

    const isKeyboardSelection =

      event.shiftKey ||

      (
        event.ctrlKey &&
        event.key.toLowerCase() === 'a'
      );


    if (!isKeyboardSelection) {

      return;

    }


    setTimeout(
      () => {

        const selectedText =
          getSelectedText();


        if (
          selectedText &&
          selectedText.length >= 2
        ) {

          saveSelectedTextToStorage(
            selectedText
          );

        }

      },
      50
    );

  }
);


// ============================================================
// MESSAGE HANDLER
// ============================================================

chrome.runtime.onMessage.addListener(
  (
    message,
    sender,
    sendResponse
  ) => {


    if (!message) {

      sendResponse({
        success: false,
        error: 'Invalid message.'
      });

      return;

    }


    // ========================================================
    // GET COMPLETE PAGE INFORMATION
    // ========================================================

    if (

      message.type ===
      'GET_PAGE_INFO' ||

      message.action ===
      'GET_PAGE_INFO'

    ) {

      sendResponse(
        getPageInfo()
      );

      return;

    }


    // ========================================================
    // GET SELECTED TEXT
    // ========================================================

    if (

      message.type ===
      'GET_SELECTED_TEXT' ||

      message.action ===
      'GET_SELECTED_TEXT'

    ) {

      sendResponse({

        success: true,

        selectedText:
          getSelectedText(),

        sourceUrl:
          window.location.href,

        sourceTitle:
          document.title

      });

      return;

    }


    // ========================================================
    // GET PAGE IMAGES
    // ========================================================

    if (

      message.type ===
      'GET_PAGE_IMAGES' ||

      message.action ===
      'GET_PAGE_IMAGES'

    ) {

      sendResponse({

        success: true,

        images:
          getPageImages()

      });

      return;

    }


    // ========================================================
    // GET PAGE TEXT
    // ========================================================

    if (

      message.type ===
      'GET_PAGE_TEXT' ||

      message.action ===
      'GET_PAGE_TEXT'

    ) {

      sendResponse({

        success: true,

        pageText:
          getPageText()
            .slice(
              0,
              MAX_PAGE_TEXT_LENGTH
            )

      });

      return;

    }


    // ========================================================
    // RIGHT-CLICK TEXT VERIFICATION
    // ========================================================

    if (

      message.action ===
      'TEXT_SELECTED_FOR_VERIFICATION'

    ) {

      const selectedText =
        (
          message.text ||
          ''
        )
          .trim();


      if (selectedText) {

        saveSelectedTextToStorage(
          selectedText
        );

      }


      sendResponse({

        success: true

      });

      return;

    }


    // ========================================================
    // PING
    // ========================================================

    if (

      message.action ===
      'PING'

    ) {

      sendResponse({

        success: true,

        message:
          'Content script is active.'

      });

      return;

    }


    // ========================================================
    // UNKNOWN MESSAGE
    // ========================================================

    sendResponse({

      success: false,

      error:
        'Unknown message type.'

    });

  }
);


// ============================================================
// CONTENT SCRIPT READY
// ============================================================

console.log(
  'Multimodal Fake Content Detector content script loaded.'
);