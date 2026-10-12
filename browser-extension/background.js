// ============================================================
// BACKGROUND SERVICE WORKER
// Multimodal Fake Content Detection Browser Extension
// ============================================================


// ============================================================
// BACKEND API CONFIGURATION
// ============================================================

const API_BASE_URL =
  'http://127.0.0.1:8000'


const BACKEND_HEALTH_URL =
  `${API_BASE_URL}/health`


// ============================================================
// BACKEND STATUS CACHE
// ============================================================

let backendOnline =
  false


let lastBackendCheck =
  0


const BACKEND_CHECK_CACHE_DURATION =
  5000


// ============================================================
// CHECK BACKEND STATUS
// ============================================================

async function checkBackend(
  forceCheck = false
) {

  const now =
    Date.now()


  // ==========================================================
  // USE RECENT STATUS CACHE
  // ==========================================================

  if (
    !forceCheck &&
    now - lastBackendCheck <
    BACKEND_CHECK_CACHE_DURATION
  ) {

    return backendOnline

  }


  let controller =
    null


  let timeout =
    null


  try {

    controller =
      new AbortController()


    timeout =
      setTimeout(
        () => {

          controller.abort()

        },
        5000
      )


    const response =
      await fetch(
        BACKEND_HEALTH_URL,
        {
          method:
            'GET',

          signal:
            controller.signal,

          cache:
            'no-store',
        }
      )


    clearTimeout(
      timeout
    )


    backendOnline =
      response.ok


    lastBackendCheck =
      Date.now()


    if (
      backendOnline
    ) {

      console.log(
        'Backend is running.'
      )

    }

    else {

      console.warn(
        `Backend responded with status: ${response.status}`
      )

    }


    return backendOnline

  }

  catch (
    error
  ) {

    if (
      timeout
    ) {

      clearTimeout(
        timeout
      )

    }


    backendOnline =
      false


    lastBackendCheck =
      Date.now()


    console.warn(
      'Backend is currently unavailable.'
    )


    return false

  }

}


// ============================================================
// CREATE RIGHT-CLICK CONTEXT MENU
// ============================================================

function createContextMenu() {

  chrome.contextMenus.removeAll(
    () => {


      if (
        chrome.runtime.lastError
      ) {

        console.warn(
          'Failed to remove old context menus:',
          chrome.runtime.lastError.message
        )

      }


      chrome.contextMenus.create(
        {
          id:
            'verify-selected-content',

          title:
            'Verify Selected Content',

          contexts:
            [
              'selection'
            ],
        },
        () => {


          if (
            chrome.runtime.lastError
          ) {

            console.warn(
              'Context menu creation error:',
              chrome.runtime.lastError.message
            )

            return

          }


          console.log(
            'Context menu created successfully.'
          )

        }
      )

    }
  )

}


// ============================================================
// EXTENSION INSTALLED OR UPDATED
// ============================================================

chrome.runtime.onInstalled.addListener(
  async () => {


    console.log(
      'Multimodal Fake Content Detection extension installed/updated.'
    )


    // ========================================================
    // CREATE CONTEXT MENU
    // ========================================================

    createContextMenu()


    // ========================================================
    // CHECK BACKEND STATUS
    // ========================================================

    await checkBackend(
      true
    )

  }
)


// ============================================================
// CHROME STARTUP
// ============================================================

chrome.runtime.onStartup.addListener(
  async () => {


    console.log(
      'Chrome started. Initializing extension.'
    )


    // ========================================================
    // CREATE CONTEXT MENU
    // ========================================================

    createContextMenu()


    // ========================================================
    // CHECK BACKEND STATUS
    // ========================================================

    await checkBackend(
      true
    )

  }
)


// ============================================================
// HANDLE RIGHT-CLICK CONTEXT MENU
// ============================================================

chrome.contextMenus.onClicked.addListener(
  async (
    info,
    tab
  ) => {


    // ========================================================
    // VERIFY CORRECT MENU ITEM
    // ========================================================

    if (
      !info ||
      info.menuItemId !==
      'verify-selected-content'
    ) {

      return

    }


    // ========================================================
    // GET SELECTED TEXT
    // ========================================================

    const selectedText =
      (
        info.selectionText ||
        ''
      )
        .trim()


    // ========================================================
    // VALIDATE SELECTED TEXT
    // ========================================================

    if (
      !selectedText
    ) {

      console.warn(
        'No text was selected.'
      )

      return

    }


    try {


      // ======================================================
      // STORE SELECTED TEXT
      // ======================================================

      await chrome.storage.local.set(
        {
          selectedText:
            selectedText,

          selectedTextTimestamp:
            Date.now(),

          selectedTextSourceUrl:
            tab?.url ||
            '',

          selectedTextSourceTitle:
            tab?.title ||
            '',
        }
      )


      console.log(
        'Selected content stored successfully.'
      )


      // ======================================================
      // IMPORTANT
      // ======================================================
      //
      // We DO NOT send chrome.tabs.sendMessage() here.
      //
      // The popup reads the selected text directly from
      // chrome.storage.local.
      //
      // This avoids:
      //
      // "Could not establish connection.
      // Receiving end does not exist."
      //
      // ======================================================


    }

    catch (
      error
    ) {

      console.error(
        'Failed to process selected text:',
        error
      )

    }

  }
)


// ============================================================
// HANDLE MESSAGES FROM POPUP OR CONTENT SCRIPT
// ============================================================

chrome.runtime.onMessage.addListener(
  (
    message,
    sender,
    sendResponse
  ) => {


    // ========================================================
    // VALIDATE MESSAGE
    // ========================================================

    if (
      !message ||
      !message.action
    ) {

      return

    }


    // ========================================================
    // BACKEND STATUS REQUEST
    // ========================================================

    if (
      message.action ===
      'CHECK_BACKEND_STATUS'
    ) {


      checkBackend(
        Boolean(
          message.forceCheck
        )
      )
        .then(
          (
            isOnline
          ) => {

            sendResponse(
              {
                success:
                  true,

                backendOnline:
                  isOnline,

                apiBaseUrl:
                  API_BASE_URL,
              }
            )

          }
        )
        .catch(
          (
            error
          ) => {

            console.error(
              'Backend status check failed:',
              error
            )


            sendResponse(
              {
                success:
                  false,

                backendOnline:
                  false,

                apiBaseUrl:
                  API_BASE_URL,
              }
            )

          }
        )


      return true

    }


    // ========================================================
    // GET STORED SELECTED TEXT
    // ========================================================

    if (
      message.action ===
      'GET_SELECTED_TEXT'
    ) {


      chrome.storage.local.get(
        [
          'selectedText',
          'selectedTextTimestamp',
          'selectedTextSourceUrl',
          'selectedTextSourceTitle',
        ],
        (
          data
        ) => {


          sendResponse(
            {
              success:
                true,

              selectedText:
                data.selectedText ||
                '',

              timestamp:
                data.selectedTextTimestamp ||
                null,

              sourceUrl:
                data.selectedTextSourceUrl ||
                '',

              sourceTitle:
                data.selectedTextSourceTitle ||
                '',
            }
          )

        }
      )


      return true

    }


    // ========================================================
    // SAVE SELECTED TEXT
    // ========================================================

    if (
      message.action ===
      'SAVE_SELECTED_TEXT'
    ) {


      const text =
        (
          message.text ||
          ''
        )
          .trim()


      chrome.storage.local.set(
        {
          selectedText:
            text,

          selectedTextTimestamp:
            Date.now(),

          selectedTextSourceUrl:
            message.sourceUrl ||
            sender?.tab?.url ||
            '',

          selectedTextSourceTitle:
            message.sourceTitle ||
            sender?.tab?.title ||
            '',
        },
        () => {


          if (
            chrome.runtime.lastError
          ) {

            sendResponse(
              {
                success:
                  false,

                error:
                  chrome.runtime.lastError.message,
              }
            )

            return

          }


          sendResponse(
            {
              success:
                true,
            }
          )

        }
      )


      return true

    }


    // ========================================================
    // CLEAR STORED SELECTED TEXT
    // ========================================================

    if (
      message.action ===
      'CLEAR_SELECTED_TEXT'
    ) {


      chrome.storage.local.remove(
        [
          'selectedText',
          'selectedTextTimestamp',
          'selectedTextSourceUrl',
          'selectedTextSourceTitle',
        ],
        () => {


          if (
            chrome.runtime.lastError
          ) {

            sendResponse(
              {
                success:
                  false,

                error:
                  chrome.runtime.lastError.message,
              }
            )

            return

          }


          sendResponse(
            {
              success:
                true,
            }
          )

        }
      )


      return true

    }


    // ========================================================
    // GET API CONFIGURATION
    // ========================================================

    if (
      message.action ===
      'GET_API_CONFIG'
    ) {


      sendResponse(
        {
          success:
            true,

          apiBaseUrl:
            API_BASE_URL,

          healthUrl:
            BACKEND_HEALTH_URL,
        }
      )


      return

    }

  }
)


// ============================================================
// SERVICE WORKER STARTED
// ============================================================

console.log(
  'Background service worker loaded.'
)