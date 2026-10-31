import re
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# model setup
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    timeout=120,
    max_retries=5,
    max_tokens=1800,
)

# 1st agent
def build_search_agent():
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="Call web_search exactly once, then summarize the results with titles and URLs. Do not call any tool again.",
    )
    return agent.with_config({"recursion_limit": 6})

# 2nd agent (no loop: picks URL in code, scrapes once, summarizes once)
class _Msg:
    def __init__(self, content):
        self.content = content

class SimpleReader:
    def invoke(self, inputs):
        text = inputs["messages"][0][1]
        urls = re.findall(r"https?://[^\s\)\]\'\"<>]+", text)
        scraped = ""
        used = ""
        for url in urls[:3]:
            out = scrape_url.invoke({"url": url})
            if out and not out.startswith("Could not scrape") and len(out) > 200:
                scraped, used = out, url
                break
        if not scraped:
            return {"messages": [_Msg("No page could be scraped. Use the search results only.")]}
        summary = llm.invoke(
            "Summarize the key facts from this web page in 8-10 bullet points.\n\n"
            f"URL: {used}\n\nContent:\n{scraped[:3000]}"
        ).content
        return {"messages": [_Msg(f"Source: {used}\n\n{summary}")]}

def build_reader_agent():
    return SimpleReader()


# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()