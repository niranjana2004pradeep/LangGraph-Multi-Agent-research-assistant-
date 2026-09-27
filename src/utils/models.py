from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter( 
    requests_per_second=0.05, 
    check_every_n_seconds=0.1, 
    max_bucket_size=10 
)

load_dotenv()
llm = ChatGroq( model="openai/gpt-oss-120b", 
               groq_api_key=os.getenv("API_KEY"), 
               rate_limiter=rate_limiter, 
               temperature=0 
)