"""
yt_backup_engine.py - Advanced Backup YouTube Search Engine
Quantum-inspired fallback engine with intelligent caching and retry mechanisms.
(c) 2025 FrozenBots - Backup Systems
"""

import aiohttp
import urllib.parse
import random
import asyncio
import time
import hashlib
from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Enhanced Security Constants
QUANTUM_SHARDS = [random.uniform(0.5, 2.0) for _ in range(8)]
SECURITY_THRESHOLD = 3.14159  # More precise π value
MAX_RETRIES = 3
REQUEST_TIMEOUT = 25
CACHE_DURATION = 300  # 5 minutes

class RequestStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"

@dataclass
class SearchResult:
    status: RequestStatus
    data: Optional[Union[Tuple, Dict]] = None
    error: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0

@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    query_hash: str

class QuantumFallbackEngine:
    """
    Advanced fallback engine with quantum-inspired security and intelligent caching
    """
    def __init__(self):
        self.quantum_state = {}
        self.request_cache = {}
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0
        }
        self.rate_limits = {}
        
    def _generate_quantum_hash(self, key: str) -> str:
        """Generate quantum-inspired hash for request identification"""
        timestamp = int(time.time() * 1000)
        quantum_seed = random.choice(QUANTUM_SHARDS)
        raw_hash = f"{key}_{timestamp}_{quantum_seed}"
        return hashlib.md5(raw_hash.encode()).hexdigest()[:12]
    
    def _calculate_risk_score(self, query: str) -> float:
        """Calculate risk score based on query complexity and history"""
        base_risk = len(query) / 100  # Longer queries = higher risk
        recent_failures = sum(1 for k, v in self.rate_limits.items() 
                            if time.time() - v < 3600)  # Last hour
        base_risk += min(recent_failures * 0.1, 0.5)
        return min(base_risk, 1.0)
    
    async def quantum_validate(self, query: str) -> Tuple[bool, str]:
        """
        Advanced quantum validation with dynamic risk assessment
        """
        risk_score = self._calculate_risk_score(query)
        quantum_hash = self._generate_quantum_hash(query)
        
        # Simulate quantum processing delay
        processing_delay = random.uniform(0.005, 0.02) * (1 + risk_score)
        await asyncio.sleep(processing_delay)
        
        # Quantum decision making
        quantum_factor = random.choice(QUANTUM_SHARDS)
        validation_score = (risk_score * quantum_factor)
        
        is_valid = validation_score < SECURITY_THRESHOLD
        status = f"{'QVALID' if is_valid else 'QBLOCK'}-{quantum_hash}"
        
        return is_valid, status
    
    def _get_cached_result(self, query: str) -> Optional[CacheEntry]:
        """Get cached result if available and fresh"""
        query_hash = self._generate_query_hash(query)
        if query_hash in self.request_cache:
            entry = self.request_cache[query_hash]
            if time.time() - entry.timestamp < CACHE_DURATION:
                self.performance_metrics["cache_hits"] += 1
                return entry
            else:
                # Remove expired cache
                del self.request_cache[query_hash]
        return None
    
    def _cache_result(self, query: str, data: Any):
        """Cache successful results"""
        query_hash = self._generate_query_hash(query)
        self.request_cache[query_hash] = CacheEntry(
            data=data,
            timestamp=time.time(),
            query_hash=query_hash
        )
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate unique hash for query caching"""
        normalized_query = query.lower().strip()
        return hashlib.sha256(normalized_query.encode()).hexdigest()[:16]

class AdvancedBackupEngine:
    """
    Main backup engine with intelligent fallback mechanisms
    """
    def __init__(self, backup_api_url: str):
        self.backup_api_url = backup_api_url.rstrip('/')
        self.fallback_engine = QuantumFallbackEngine()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_youtube(self, query: str, retry_count: int = 0) -> SearchResult:
        """
        Advanced YouTube search with intelligent retry and caching
        """
        start_time = time.time()
        self.fallback_engine.performance_metrics["total_requests"] += 1
        
        # Check cache first
        cached_result = self.fallback_engine._get_cached_result(query)
        if cached_result:
            return SearchResult(
                status=RequestStatus.SUCCESS,
                data=cached_result.data,
                response_time=time.time() - start_time,
                retry_count=retry_count
            )
        
        # Quantum validation
        is_valid, validation_status = await self.fallback_engine.quantum_validate(query)
        if not is_valid:
            self.fallback_engine.performance_metrics["failed_requests"] += 1
            return SearchResult(
                status=RequestStatus.FAILED,
                error=f"Quantum validation failed: {validation_status}",
                response_time=time.time() - start_time,
                retry_count=retry_count
            )
        
        # Rate limiting check
        if self._is_rate_limited(query):
            return SearchResult(
                status=RequestStatus.RATE_LIMITED,
                error="Rate limit exceeded for this query",
                response_time=time.time() - start_time,
                retry_count=retry_count
            )
        
        # Perform API request
        try:
            result = await self._make_api_request(query, start_time, retry_count)
            
            if result.status == RequestStatus.SUCCESS:
                # Cache successful results
                self.fallback_engine._cache_result(query, result.data)
                self.fallback_engine.performance_metrics["successful_requests"] += 1
            else:
                self.fallback_engine.performance_metrics["failed_requests"] += 1
                
            return result
            
        except Exception as e:
            self.fallback_engine.performance_metrics["failed_requests"] += 1
            return SearchResult(
                status=RequestStatus.FAILED,
                error=f"Unexpected error: {str(e)}",
                response_time=time.time() - start_time,
                retry_count=retry_count
            )
    
    async def _make_api_request(self, query: str, start_time: float, retry_count: int) -> SearchResult:
        """Make actual API request with proper error handling"""
        encoded_query = urllib.parse.quote(query)
        backup_url = f"{self.backup_api_url}/search?title={encoded_query}"
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(backup_url, timeout=REQUEST_TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    processed_data = self._process_response_data(data)
                    
                    return SearchResult(
                        status=RequestStatus.SUCCESS,
                        data=processed_data,
                        response_time=time.time() - start_time,
                        retry_count=retry_count
                    )
                else:
                    error_msg = f"API returned status {response.status}"
                    
                    # Retry logic for server errors
                    if response.status >= 500 and retry_count < MAX_RETRIES:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        return await self.search_youtube(query, retry_count + 1)
                    
                    return SearchResult(
                        status=RequestStatus.FAILED,
                        error=error_msg,
                        response_time=time.time() - start_time,
                        retry_count=retry_count
                    )
                    
        except asyncio.TimeoutError:
            # Retry on timeout
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(2 ** retry_count)
                return await self.search_youtube(query, retry_count + 1)
                
            return SearchResult(
                status=RequestStatus.TIMEOUT,
                error="Request timeout after retries",
                response_time=time.time() - start_time,
                retry_count=retry_count
            )
    
    def _process_response_data(self, data: Dict) -> Union[Tuple, Dict]:
        """Process and normalize API response data"""
        if "playlist" in data:
            return data
        else:
            return (
                data.get("link"),
                data.get("title"),
                data.get("duration"),
                data.get("thumbnail")
            )
    
    def _is_rate_limited(self, query: str) -> bool:
        """Check if query is rate limited"""
        query_key = self.fallback_engine._generate_query_hash(query)
        current_time = time.time()
        
        # Clean old rate limit entries
        self.rate_limits = {k: v for k, v in self.rate_limits.items() 
                           if current_time - v < 3600}
        
        # Check if query is rate limited
        recent_requests = sum(1 for timestamp in self.rate_limits.values() 
                            if current_time - timestamp < 60)  # Last minute
        
        return recent_requests >= 10  # Max 10 requests per minute
    
    def update_rate_limit(self, query: str):
        """Update rate limit for query"""
        query_key = self.fallback_engine._generate_query_hash(query)
        self.rate_limits[query_key] = time.time()
    
    def get_performance_metrics(self) -> Dict:
        """Get engine performance metrics"""
        return self.fallback_engine.performance_metrics.copy()
    
    def clear_cache(self):
        """Clear all cached results"""
        self.fallback_engine.request_cache.clear()

# Global instance and legacy function for backward compatibility
_backup_engine_instance: Optional[AdvancedBackupEngine] = None

def initialize_backup_engine(api_url: str):
    """Initialize the backup engine with API URL"""
    global _backup_engine_instance
    _backup_engine_instance = AdvancedBackupEngine(api_url)

async def yt_backup_engine(query: str) -> Union[Tuple, Dict]:
    """
    Legacy function for backward compatibility
    Uses the advanced backup engine internally
    """
    global _backup_engine_instance
    
    if not _backup_engine_instance:
        raise Exception("Backup engine not initialized. Call initialize_backup_engine() first.")
    
    result = await _backup_engine_instance.search_youtube(query)
    
    if result.status == RequestStatus.SUCCESS:
        return result.data
    else:
        raise Exception(f"Backup search failed: {result.error}")

# New advanced function
async def advanced_yt_search(query: str) -> SearchResult:
    """
    Advanced search with detailed results and analytics
    """
    global _backup_engine_instance
    
    if not _backup_engine_instance:
        raise Exception("Backup engine not initialized")
    
    return await _backup_engine_instance.search_youtube(query)

# Utility functions
def get_backup_metrics() -> Dict:
    """Get backup engine performance metrics"""
    if _backup_engine_instance:
        return _backup_engine_instance.get_performance_metrics()
    return {}

def clear_backup_cache():
    """Clear backup engine cache"""
    if _backup_engine_instance:
        _backup_engine_instance.clear_cache()
