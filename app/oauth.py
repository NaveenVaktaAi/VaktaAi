import time
from datetime import datetime

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwe, jwt
from app.database.session import get_db
from app.database.mongo_collections import get_user
from app.common import constants
from app.features.auth.schema import CurrentUser
from pymongo.database import Database

env_data = {
    "SECRET_KEY": "vakta_ai_secret_id",
    "ALGORITHM": "HS256"
}

security = HTTPBearer()


async def is_user_authorized(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Database = Depends(get_db),
) -> dict:
    try:
        token = credentials.credentials
        jwt_token = jwe.decrypt(token, "agent_ai_secret_id".ljust(16)[:16] )

        payload = jwt.decode(
            jwt_token, env_data.get("SECRET_KEY"), algorithms=env_data.get("ALGORITHM")
        )

        current_time = int(time.time())
        now = datetime.utcnow()

        id = payload.get("id")
        email = payload.get("email")
        user_type = payload.get("role_type")

        if not id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": constants.ACCESS_DENIED},
            )

        user = get_user(db, id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": constants.USER_NOT_FOUND},
            )
        
        if user.is_delete:
            raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"error": "Your account is removed. Please contact support."},
                )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": constants.SUPERUSER_INACTIVE},
            )
 
        # if token_from_redis is None:
        #     raise HTTPException(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         detail={"error": constants.TOKEN_NOT_FOUND},
        #     )

        if payload["exp"] <= current_time:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": constants.SESSION_EXPIRED},
            )

        if now > datetime.fromtimestamp(payload.get("exp")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": constants.SESSION_EXPIRED},
            )

        return {"id": id, "email": email, "role_type": user_type}
    except Exception as e:
        print(e, "Exception")
        error_message = str(e)
        detail = constants.INVALID_TOKEN

        if error_message in [
            constants.SUPERUSER_INACTIVE,
            constants.ACCESS_DENIED,
            constants.EMAIL_OR_PASSWORD_INCORRECT,
            constants.SUPERUSER_NOT_VERIFIED,
            constants.TOKEN_NOT_FOUND,
            constants.SESSION_EXPIRED,
        ]:
            detail = error_message

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": detail},
        )


async def is_admin_authorized(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Database = Depends(get_db),
) -> CurrentUser:
    try:
        user = await is_user_authorized(request, credentials, db)
        user_type = user["role_type"]
        if user_type != constants.ROLE_TYPE["SUPER_ADMIN"]:
            raise ValueError(constants.SUPERADMIN_NOT_FOUND)
        return user
    except Exception as e:
        print(e, "Exception")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": (
                    constants.SUPERUSER_INACTIVE
                    if str(e) == constants.SUPERUSER_INACTIVE
                    else constants.INVALID_TOKEN
                )
            },
        )
