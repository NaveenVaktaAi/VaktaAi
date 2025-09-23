import ast
from urllib.parse import urlparse

# from urllib.parse import urlparse

from fastapi import APIRouter

from app.config import env_variables
from app.request import Request
from app.utils.manifest import generate_html_scripts
from app.utils.templates import templates
from app.utils.variables import MANIFEST_FILES

router = APIRouter()
env_data = env_variables()


@router.get("/widget/agent_ai")
async def get_widget(request: Request):
    """
    Retrieves JS to load the iFrame and other dynamic code.
    """

    return templates.TemplateResponse(
        "widget.js",
        {
            "request": request,
        },
        media_type="application/javascript",
    )


@router.get("/widget/agent_ai/iframe")
async def get_widget_iframe(request: Request):
    """
    Retrieves the iFrame to load the widget and sets client session.
    """
    # Don't bother with cross site security. At the end of the day, someone
    # can just inspect the network and find all socket connections and parameters
    # to replicate the chat. What we should worry about is rate limiting.

    allowed_domains = ast.literal_eval(env_data.get("ALLOWED_HOST_FOR_CHATBOT"))
    parsed_referer = urlparse(request.headers.get("referer"))
    referer_domain = parsed_referer.hostname
    print(referer_domain, "referer_domain", allowed_domains)
    if allowed_domains and referer_domain not in allowed_domains:
        not_allowed_data = {
            "request": request,
            "secure_image_url": env_data.get("SECURE_ORIGIN_IMAGE"),
        }
        return templates.TemplateResponse("not_allowed.html", not_allowed_data)

    default_kwargs = {
        "request": request,
        "event_allowed_host": allowed_domains,
    }

    if env_data.get("LOAD_MANIFEST").lower() == "true":
        manifest = MANIFEST_FILES.get()["chatbot"]
        default_kwargs["html_scripts"] = generate_html_scripts(
            manifest, project="chatbot"
        )

    return templates.TemplateResponse("widget_iframe.html", default_kwargs)


@router.get("/widget/agent_ai/admin")
async def get_widget(request: Request):
    """
    Retrieves JS to load the iFrame and other dynamic code.
    """

    return templates.TemplateResponse(
        "widget_admin.js",
        {
            "request": request,
        },
        media_type="application/javascript",
    )


@router.get("/widget/agent_ai/iframe/admin/upload-document")
async def get_widget_iframe_admin(request: Request):
    """
    Retrieves the iFrame to load the widget and sets client session.
    """
    # Don't bother with cross site security. At the end of the day, someone
    # can just inspect the network and find all socket connections and parameters
    # to replicate the chat. What we should worry about is rate limiting.

    allowed_domains = ast.literal_eval(env_data.get("ALLOWED_HOST_FOR_CHATBOT"))
    parsed_referer = urlparse(request.headers.get("referer"))
    referer_domain = parsed_referer.hostname
    if allowed_domains and referer_domain not in allowed_domains:
        not_allowed_data = {
            "request": request,
            "secure_image_url": env_data.get("SECURE_ORIGIN_IMAGE"),
        }
        return templates.TemplateResponse("not_allowed.html", not_allowed_data)

    default_kwargs = {
        "request": request,
        "event_allowed_host": allowed_domains,
    }

    if env_data.get("LOAD_MANIFEST").lower() == "true":
        manifest = MANIFEST_FILES.get()["chatbot"]
        print(manifest, "manifestmanifestmanifestmanifest")
        p = generate_html_scripts(
            manifest, project="chatbot"
        )
        print(manifest, "ppppppppppppppppppppppp",p)
        default_kwargs["html_scripts"]=p

    return templates.TemplateResponse("widget_iframe_admin.html", default_kwargs)


@router.get("/widget/agent_ai/iframe/admin/question-listing")
async def get_widget_iframe_admin_question_listing(request: Request):
    """
    Retrieves the iFrame to load the widget and sets client session.
    """
    # Don't bother with cross site security. At the end of the day, someone
    # can just inspect the network and find all socket connections and parameters
    # to replicate the chat. What we should worry about is rate limiting.

    allowed_domains = ast.literal_eval(env_data.get("ALLOWED_HOST_FOR_CHATBOT"))
    parsed_referer = urlparse(request.headers.get("referer"))
    referer_domain = parsed_referer.hostname
    if allowed_domains and referer_domain not in allowed_domains:
        not_allowed_data = {
            "request": request,
            "secure_image_url": env_data.get("SECURE_ORIGIN_IMAGE"),
        }
        return templates.TemplateResponse("not_allowed.html", not_allowed_data)

    default_kwargs = {
        "request": request,
        "event_allowed_host": allowed_domains,
    }

    if env_data.get("LOAD_MANIFEST").lower() == "true":
        manifest = MANIFEST_FILES.get()["chatbot"]
        default_kwargs["html_scripts"] = generate_html_scripts(
            manifest, project="chatbot"
        )

    return templates.TemplateResponse(
        "widget_iframe_admin_question_listing.html", default_kwargs
    )


@router.get("/widget/agent_ai/admin/question-listing")
async def get_widget(request: Request):
    """
    Retrieves JS to load the iFrame and other dynamic code.
    """

    return templates.TemplateResponse(
        "widget_admin_questions_listing.js",
        {
            "request": request,
        },
        media_type="application/javascript",
    )


@router.get("/widget/agent_ai/admin/chat-history")
async def get_widget(request: Request):
    """
    Retrieves JS to load the iFrame and other dynamic code.
    """

    return templates.TemplateResponse(
        "widget_chat_history.js",
        {
            "request": request,
        },
        media_type="application/javascript",
    )


@router.get("/widget/agent_ai/iframe/admin/chat-history")
async def get_widget_iframe_admin(request: Request):
    """
    Retrieves the iFrame to load the widget and sets client session.
    """
    # Don't bother with cross site security. At the end of the day, someone
    # can just inspect the network and find all socket connections and parameters
    # to replicate the chat. What we should worry about is rate limiting.

    allowed_domains = ast.literal_eval(env_data.get("ALLOWED_HOST_FOR_CHATBOT"))
    parsed_referer = urlparse(request.headers.get("referer"))
    referer_domain = parsed_referer.hostname
    if allowed_domains and referer_domain not in allowed_domains:
        not_allowed_data = {
            "request": request,
            "secure_image_url": env_data.get("SECURE_ORIGIN_IMAGE"),
        }
        return templates.TemplateResponse("not_allowed.html", not_allowed_data)

    default_kwargs = {
        "request": request,
        "event_allowed_host": allowed_domains,
    }

    if env_data.get("LOAD_MANIFEST").lower() == "true":
        manifest = MANIFEST_FILES.get()["chatbot"]
        default_kwargs["html_scripts"] = generate_html_scripts(
            manifest, project="chatbot"
        )

    return templates.TemplateResponse("widget_iframe_chat_history.html", default_kwargs)
