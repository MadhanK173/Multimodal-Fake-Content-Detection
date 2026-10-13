from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.prediction import router as prediction_router
from .api.verification import router as verification_router
from .plugins.plugin_manager import PluginManager


app = FastAPI(
    title="Multimodal Fake Content Detection API",
    description=(
        "Backend API for detecting fake content "
        "across text, image, and audio modalities."
    ),
    version="1.0.0"
)


# ============================================================
# PLUGIN MANAGEMENT
# ============================================================

plugin_manager = PluginManager()


@app.get("/api/v1/plugins")
async def get_plugins():
    """
    Return all registered plugins.
    """

    plugins = []

    for name, plugin in plugin_manager.list_plugins().items():
        plugins.append({
            "name": name,
            "class": plugin.__class__.__name__,
            "modality": getattr(
                plugin,
                "modality",
                name
            ),
            "status": "active",
        })

    return {
        "status": "success",
        "plugin_count": len(plugins),
        "plugins": plugins,
    }


@app.get("/api/v1/plugins/{plugin_name}")
async def get_plugin(plugin_name: str):
    """
    Return information about a specific registered plugin.
    """

    if not plugin_manager.has_plugin(plugin_name):
        return {
            "status": "error",
            "message": (
                f"Plugin '{plugin_name}' "
                "is not registered."
            ),
        }

    plugin = plugin_manager.get_plugin(
        plugin_name
    )

    return {
        "status": "success",
        "plugin": {
            "name": plugin_name,
            "class": plugin.__class__.__name__,
            "modality": getattr(
                plugin,
                "modality",
                plugin_name
            ),
            "status": "active",
        },
    }


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://multimodal-fake-content-detection.vercel.app",
    ],

    # Allow Chrome extension origins dynamically.
    allow_origin_regex=r"^chrome-extension://[a-z]{32}$",

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ],

    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "message": (
            "Multimodal Fake Content Detection API "
            "is running successfully"
        ),
        "status": "online"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "backend"
    }


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get("/api/v1/system/info")
async def system_info():
    return {
        "application": "Multimodal Fake Content Detection System",
        "version": "1.0.0",
        "supported_modalities": [
            "text",
            "image",
            "audio"
        ],
        "status": "development"
    }


# ============================================================
# PREDICTION ROUTES
# ============================================================

app.include_router(
    prediction_router
)

app.include_router(
    verification_router
)