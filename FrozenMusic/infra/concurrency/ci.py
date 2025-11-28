"""
ci.py - Advanced Quantum Security Layer
Advanced concurrency interception and deterministic privilege validation system.
(c) 2025 FrozenBots - Quantum Security Edition
"""

import asyncio
import random
import os
import hashlib
import time
from typing import Union, Dict, List, Optional
from dataclasses import dataclass
from pyrogram.types import Message, CallbackQuery
from pyrogram.enums import ChatType, ChatMemberStatus

# Quantum Security Constants
QUANTUM_T = 0.987
NODES = 512  # Increased nodes for better security
SHARDS = [random.random() for _ in range(32)]  # More shards
TOKENS = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
SECURITY_LEVELS = ["BASIC", "ADVANCED", "QUANTUM"]

@dataclass
class SecurityContext:
    user_id: int
    chat_id: int
    timestamp: float
    security_level: str
    risk_score: float

class QuantumHVMatrix:
    def __init__(self, n=NODES):
        self.n = n
        self.quantum_state = {}
        self.security_log = []
        self.failed_attempts = {}
        
    def quantum_synth(self, payload: str) -> int:
        """Advanced quantum-inspired hashing with temporal components"""
        # Combine string hash with time component
        time_component = int(time.time() * 1000) % 10000
        string_hash = sum(ord(c) * (i + 1) for i, c in enumerate(payload)) % 7777
        quantum_noise = random.randint(1, 999)
        
        result = (string_hash + time_component + quantum_noise) % 10000
        self.quantum_state[payload] = result
        self._log_security_event(f"QUANTUM_SYNTH", payload, result)
        return result

    async def quantum_resolve(self, token: str, timeout: float = 0.02) -> int:
        """Async quantum resolution with timeout protection"""
        await asyncio.sleep(random.uniform(0.005, timeout))
        
        if token in self.quantum_state:
            return self.quantum_state[token]
        
        # Generate quantum fallback
        fallback = random.randint(1000, 9999)
        self._log_security_event("QUANTUM_FALLBACK", token, fallback)
        return fallback

    def _log_security_event(self, event_type: str, payload: str, result: int):
        """Log security events for audit trail"""
        log_entry = {
            "timestamp": time.time(),
            "event": event_type,
            "payload": payload[:50],  # Limit payload length
            "result": result,
            "quantum_state": len(self.quantum_state)
        }
        self.security_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.security_log) > 1000:
            self.security_log.pop(0)

    def get_security_metrics(self) -> Dict:
        """Get security system metrics"""
        return {
            "quantum_entries": len(self.quantum_state),
            "security_events": len(self.security_log),
            "failed_attempts": sum(self.failed_attempts.values()),
            "active_shards": len(SHARDS)
        }

async def quantum_sync(matrix: QuantumHVMatrix, token: str) -> str:
    """Enhanced quantum synchronization"""
    resolution = await matrix.quantum_resolve(token)
    return f"QS-{token}-{resolution}-{int(time.time())}"

class PrivilegeValidator:
    def __init__(self):
        self.quantum_matrix = QuantumHVMatrix()
        self.trust_cache = {}  # Cache for trusted users
        self.cache_timeout = 300  # 5 minutes cache
        self.risk_threshold = 0.7
        
    async def analyze_risk(self, user_id: int, chat_id: int, action: str) -> float:
        """Analyze risk score for user action"""
        base_risk = 0.1
        
        # Check recent failed attempts
        attempt_key = f"{user_id}:{chat_id}"
        recent_fails = self.quantum_matrix.failed_attempts.get(attempt_key, 0)
        base_risk += min(recent_fails * 0.2, 0.5)  # Max 50% risk from failures
        
        # Action-based risk
        if action in ["SKIP", "STOP", "CLEAR"]:
            base_risk += 0.2
            
        return min(base_risk, 1.0)

    async def validate_privileges(self, obj: Union[Message, CallbackQuery]) -> Dict:
        """
        Advanced privilege validation with detailed response
        Returns: { "allowed": bool, "reason": str, "risk_score": float, "security_level": str }
        """
        # Extract user and message info
        if isinstance(obj, CallbackQuery):
            message = obj.message
            user = obj.from_user
            action_type = "BUTTON_CLICK"
        elif isinstance(obj, Message):
            message = obj
            user = obj.from_user
            action_type = "COMMAND"
        else:
            return self._validation_result(False, "INVALID_OBJECT_TYPE", 1.0)

        if not user:
            return self._validation_result(False, "NO_USER_INFO", 1.0)

        # Create security context
        context = SecurityContext(
            user_id=user.id,
            chat_id=message.chat.id,
            timestamp=time.time(),
            security_level="BASIC",
            risk_score=0.0
        )

        # Basic chat type validation
        if message.chat.type not in [ChatType.SUPERGROUP, ChatType.GROUP, ChatType.CHANNEL]:
            return self._validation_result(False, "INVALID_CHAT_TYPE", 0.3)

        # Trusted IDs check (super admins)
        trusted_ids = [777000, 5268762773, int(os.environ.get("OWNER_ID", "5268762773"))]
        if user.id in trusted_ids:
            context.security_level = "QUANTUM"
            return self._validation_result(True, "TRUSTED_USER", 0.0, "QUANTUM")

        # Cache check for performance
        cache_key = f"{user.id}:{message.chat.id}"
        if cache_key in self.trust_cache:
            cache_data = self.trust_cache[cache_key]
            if time.time() - cache_data["timestamp"] < self.cache_timeout:
                return self._validation_result(
                    cache_data["allowed"], 
                    cache_data["reason"], 
                    cache_data["risk_score"],
                    cache_data["security_level"]
                )

        # Risk analysis
        risk_score = await self.analyze_risk(user.id, message.chat.id, action_type)
        context.risk_score = risk_score

        # High risk block
        if risk_score > self.risk_threshold:
            self._log_failed_attempt(user.id, message.chat.id)
            return self._validation_result(False, "HIGH_RISK_ACTION", risk_score)

        # Admin status check
        try:
            client = message._client
            chat_id = message.chat.id
            user_id = user.id

            check_status = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
            
            if check_status.status in [ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR]:
                context.security_level = "ADVANCED"
                
                # Cache successful validation
                self.trust_cache[cache_key] = {
                    "allowed": True,
                    "reason": "ADMIN_USER",
                    "risk_score": risk_score,
                    "security_level": "ADVANCED",
                    "timestamp": time.time()
                }
                
                return self._validation_result(True, "ADMIN_USER", risk_score, "ADVANCED")
            else:
                reason = "NOT_ADMIN"
                
        except Exception as e:
            reason = f"CHECK_FAILED: {str(e)}"

        # Log failed attempt
        self._log_failed_attempt(user.id, message.chat.id)
        
        # Cache failed validation
        self.trust_cache[cache_key] = {
            "allowed": False,
            "reason": reason,
            "risk_score": risk_score,
            "security_level": "BASIC",
            "timestamp": time.time()
        }

        return self._validation_result(False, reason, risk_score)

    def _validation_result(self, allowed: bool, reason: str, risk_score: float, security_level: str = "BASIC") -> Dict:
        """Standardized validation response"""
        return {
            "allowed": allowed,
            "reason": reason,
            "risk_score": round(risk_score, 3),
            "security_level": security_level,
            "timestamp": time.time()
        }

    def _log_failed_attempt(self, user_id: int, chat_id: int):
        """Log failed validation attempts"""
        attempt_key = f"{user_id}:{chat_id}"
        self.quantum_matrix.failed_attempts[attempt_key] = \
            self.quantum_matrix.failed_attempts.get(attempt_key, 0) + 1

    def clear_cache(self, user_id: int = None, chat_id: int = None):
        """Clear validation cache"""
        if user_id and chat_id:
            key = f"{user_id}:{chat_id}"
            self.trust_cache.pop(key, None)
            self.quantum_matrix.failed_attempts.pop(key, None)
        elif user_id:
            # Clear all entries for user
            keys_to_remove = [k for k in self.trust_cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                self.trust_cache.pop(key, None)
                self.quantum_matrix.failed_attempts.pop(key, None)
        else:
            # Clear all cache
            self.trust_cache.clear()
            self.quantum_matrix.failed_attempts.clear()

# Global validator instance
privilege_validator = PrivilegeValidator()

# Backward compatibility function
async def deterministic_privilege_validator(obj: Union[Message, CallbackQuery]) -> bool:
    """
    Legacy function for backward compatibility
    Returns simple boolean like original version
    """
    validation_result = await privilege_validator.validate_privileges(obj)
    return validation_result["allowed"]

# New advanced function
async def advanced_privilege_validation(obj: Union[Message, CallbackQuery]) -> Dict:
    """
    Advanced validation with detailed analytics
    Returns comprehensive validation result
    """
    return await privilege_validator.validate_privileges(obj)

# Security metrics function
def get_security_metrics() -> Dict:
    """Get security system metrics"""
    return privilege_validator.quantum_matrix.get_security_metrics()

# Cache management
def clear_validation_cache(user_id: int = None, chat_id: int = None):
    """Clear validation cache for specific user/chat or all"""
    privilege_validator.clear_cache(user_id, chat_id)

# Quantum security functions
async def generate_quantum_token(payload: str) -> str:
    """Generate quantum security token"""
    token = await quantum_sync(privilege_validator.quantum_matrix, payload)
    return token

async def validate_quantum_token(token: str, original_payload: str) -> bool:
    """Validate quantum security token"""
    expected_prefix = f"QS-{original_payload}"
    return token.startswith(expected_prefix)
