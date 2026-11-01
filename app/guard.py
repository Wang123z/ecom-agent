"""鉴权与限流。

JWT 只解决「你是谁」，真正的价值在于 user_id 由服务端签发、用户改不了，
可以放心当数据过滤条件——查订单时用它校验归属，防 IDOR。

限流用 Redis 令牌桶（Lua 保证原子），Redis 不可用时放行，不阻断主链路。
"""
import time

from fastapi import Depends, Header, HTTPException, Request
from jose import JWTError, jwt

from app import config

LUA_TOKEN_BUCKET = """
local key, rate, burst, now = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1]) or burst
local ts = tonumber(data[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
if tokens < 1 then return 0 end
redis.call('HMSET', key, 'tokens', tokens - 1, 'ts', now)
redis.call('EXPIRE', key, 120)
return 1
"""

_redis = None


def redis_client():
    global _redis
    if _redis is None:
        import redis

        _redis = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis


def create_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + config.JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, config.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(401, "登录已过期，请重新登录")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(401, "token 无效")
    return user_id


async def current_user(request: Request, authorization: str = Header("")) -> str:
    """从 Authorization 取 user_id，写进 request.state 供日志与下游使用。"""
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "缺少登录凭证")
    user_id = decode_token(token)
    request.state.user_id = user_id
    return user_id


def allow(user_id: str) -> bool:
    """用户级令牌桶：每分钟 RATE_LIMIT_USER_PER_MIN 次，突发 5。"""
    rate = config.RATE_LIMIT_USER_PER_MIN / 60.0
    try:
        return bool(redis_client().eval(
            LUA_TOKEN_BUCKET, 1, f"rl:user:{user_id}",
            rate, 5, time.time(),
        ))
    except Exception:
        return True          # Redis 挂了放行，限流不该拖垮服务


async def rate_limit(user_id: str = Depends(current_user)) -> str:
    """FastAPI 依赖：先鉴权再限流，超限返回 429。"""
    if not allow(user_id):
        raise HTTPException(429, "请求过于频繁，请稍后再试")
    return user_id
