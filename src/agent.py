"""
Part 3 - Flipkart Support LangGraph Agent

Default mode:
    MOCK_LLM

The default mode:
    - requires no API key
    - makes no outbound network calls
    - does not use a live LLM

The graph contains four required nodes:

1. Intent node
2. RAG retrieval node
3. Tool calling node
4. Response generation node

The graph uses conditional routing based on intent.
"""
import os

# Prevent native ML libraries from spawning conflicting threads
# on macOS when PyTorch / FAISS / sentence-transformers
# are used together.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any
import json
import re

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from .tools import (
    check_return_risk,
    classify_product_image
)


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = ROOT / "indexes" / "policy.index"
METADATA_FILE = ROOT / "indexes" / "metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum similarity required for a grounded policy answer.
#
# Based on the current retrieval evaluation:
#   valid policy queries: approximately 0.59 - 0.77
#   known ungrounded query: 0.4842
#
# 0.55 separates the current valid and ungrounded test cases.
GROUNDING_THRESHOLD = 0.55


# =========================================================
# LOAD RAG COMPONENTS
# =========================================================

print("Loading RAG embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunk_metadata = json.load(f)


# =========================================================
# GRAPH STATE
# =========================================================

class AgentState(TypedDict, total=False):

    user_input: str

    conversation_id: str

    messages: List[Dict[str, str]]

    intent: str

    retrieved_chunks: List[Dict[str, Any]]

    tool_result: Dict[str, Any]

    answer: str

    source: str

    confidence: float

    blocked: bool

    grounded: bool

    order_id: Optional[str]

    order_features: Optional[Dict[str, Any]]

    image_path: Optional[str]

    last_intent: Optional[str]

    last_result: Optional[Dict[str, Any]]

    response: Dict[str, Any]

    final_response: Dict[str, Any]


# =========================================================
# 4S + ROLE PROMPT
# =========================================================

SYSTEM_PROMPT = """

ROLE:
You are Flipkart's support assistant.

SPECIFIC:
Classify every customer request as exactly one of:

- policy
- return_risk
- product_category

SHORT:
Use only the information necessary to answer the customer.

SURROUND:
For policy questions, use only retrieved policy knowledge-base content.

For return-risk questions, use the real Part 1 return-risk model.

For product-category questions, use the real Part 2 image classifier.

SINGLE:
Return exactly one structured JSON response with:

{
    "answer": "...",
    "source": "...",
    "confidence": 0.0
}

Allowed source values:

- policy_kb
- return_risk_tool
- image_classifier_tool


FEW-SHOT INTENT EXAMPLES:

Example 1:

User:
"What is the return window for footwear?"

Intent:
policy


Example 2:

User:
"Is order 12345 likely to be returned?"

Intent:
return_risk


Example 3:

User:
"What category does this product image belong to?"

Intent:
product_category


Do not invent policy information.

"""


# =========================================================
# INPUT GUARDRAIL
# =========================================================

INJECTION_PATTERNS = [

    r"ignore previous instructions",

    r"ignore all rules",

    r"ignore the rules",

    r"forget your instructions",

    r"pretend you are",

    r"act as an unrestricted",

    r"bypass your instructions",

    r"reveal your system prompt"

]

def resolve_image_path(image_path: str) -> str:
    """
    Convert a relative sample-image path into an absolute
    project path.

    Example:

        data/sample_images/01_ankle_boot.png

    becomes:

        /.../flipkart-support-assistant/
        data/sample_images/01_ankle_boot.png
    """

    path = Path(image_path)

    # Already an absolute path
    if path.is_absolute() and path.exists():
        return str(path)

    # Relative path from project root
    project_path = ROOT / path

    if project_path.exists():
        return str(project_path)

    # Handle paths such as /sample_images/file.png
    # by looking inside data/sample_images/
    filename = path.name

    sample_path = (
        ROOT
        / "data"
        / "sample_images"
        / filename
    )

    if sample_path.exists():
        return str(sample_path)

    raise FileNotFoundError(
        f"Could not find image: {image_path}"
    )


def detect_prompt_injection(text: str) -> bool:

    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# =========================================================
# EXTRACT ORDER ID
# =========================================================

def extract_order_id(text: str):

    """
    Extract a numeric order ID from the user message.

    Example:

        "Check order 123456"

    returns:

        "123456"
    """

    match = re.search(
        r"\border\s*#?\s*(\d{4,})\b",
        text,
        flags=re.IGNORECASE
    )

    if match:

        return match.group(1)

    return None


# =========================================================
# EXTRACT IMAGE PATH
# =========================================================

def extract_image_path(text: str):

    """
    Find a PNG path mentioned in the user message.
    """

    match = re.search(
        r"(/[^\s]+\.png)",
        text,
        flags=re.IGNORECASE
    )

    if match:

        return match.group(1)

    return None


# =========================================================
# INTENT NODE
# =========================================================

def intent_node(state: AgentState):

    user_input = state["user_input"]

    # -----------------------------------------------------
    # INPUT GUARDRAIL
    # -----------------------------------------------------

    if detect_prompt_injection(user_input):

        return {

            "blocked": True,

            "intent": "blocked"

        }


    text = user_input.lower()


    # -----------------------------------------------------
    # Extract order ID
    # -----------------------------------------------------

    current_order_id = extract_order_id(
        user_input
    )

    if current_order_id:

        order_id = current_order_id

    else:

        order_id = state.get(
            "order_id"
        )


    # -----------------------------------------------------
    # PRODUCT CATEGORY
    # -----------------------------------------------------

    image_keywords = [

        "image",
        "photo",
        "picture",
        "png",
        "product category",
        "classify"

    ]

    if any(
        keyword in text
        for keyword in image_keywords
    ):

        image_path = extract_image_path(
            user_input
        )

        return {

            "intent": "product_category",

            "image_path": image_path,

            "order_id": order_id

        }


    # -----------------------------------------------------
    # RETURN RISK
    # -----------------------------------------------------

    risk_keywords = [

    "return risk",
    "likely to return",
    "likely to be returned",
    "return probability",
    "probability of return",
    "risk of return",
    "return likelihood",
    "likelihood of return",
    "risk bucket",
    "return-risk",
    "return risk score",
    "likely returned",
    "will be returned"


    ]

    if any(
        keyword in text
        for keyword in risk_keywords
    ):

        return {

            "intent": "return_risk",

            "order_id": order_id

        }


    # -----------------------------------------------------
    # FOLLOW-UP RETURN-RISK QUESTION
    #
    # Example:
    #
    # Turn 1:
    # "Check order 987654 for return risk"
    #
    # Turn 2:
    # "What risk bucket did you give that order?"
    # -----------------------------------------------------

    followup_keywords = [

        "what risk bucket",
        "which risk bucket",
        "what was the risk",
        "what risk did",
        "that order",
        "same order",
        "previous order"

    ]

    if (

        state.get("last_intent") == "return_risk"

        and

        any(
            keyword in text
            for keyword in followup_keywords
        )

    ):

        return {

            "intent": "return_risk",

            "order_id": order_id

        }


    # -----------------------------------------------------
    # DEFAULT = POLICY
    # -----------------------------------------------------

    return {

        "intent": "policy",

        "order_id": order_id

    }


# =========================================================
# RAG RETRIEVAL NODE
# =========================================================

def retrieve_policy_node(state: AgentState):

    user_input = state["user_input"]

    query_embedding = embedding_model.encode(

        [user_input],

        convert_to_numpy=True,

        normalize_embeddings=True

    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(

        query_embedding,

        3

    )

    retrieved = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue

        metadata = chunk_metadata[idx].copy()

        metadata["similarity"] = float(
            score
        )

        retrieved.append(
            metadata
        )

    return {

        "retrieved_chunks": retrieved

    }


# =========================================================
# TOOL NODE
# =========================================================

def tool_node(state: AgentState):

    intent = state["intent"]


    # =====================================================
    # RETURN RISK
    # =====================================================

    if intent == "return_risk":

        order_features = state.get(
            "order_features"
        )

        last_result = state.get(
            "last_result"
        )

        # -------------------------------------------------
        # MULTI-TURN FOLLOW-UP
        #
        # If the user asks:
        #
        # "What risk bucket did you give that order?"
        #
        # reuse the previous real model result.
        # -------------------------------------------------

        if (

            not order_features

            and

            last_result

            and

            "return_probability" in last_result

        ):

            return {

                "tool_result": last_result

            }


        # -------------------------------------------------
        # If no features exist, do not fabricate them for
        # a fresh conversation.
        # -------------------------------------------------

        if not order_features:

            return {

                "tool_result": {

                    "error":
                    "No order features are available for this conversation."

                }

            }


        # -------------------------------------------------
        # Call REAL Part 1 model
        # -------------------------------------------------

        result = check_return_risk(
            order_features
        )

        return {

            "tool_result": result

        }


    # =====================================================
    # PRODUCT IMAGE
    # =====================================================

    if intent == "product_category":

        image_path = state.get(
            "image_path"
        )


        # If the user did not specify an image,
        # use a real committed sample image.

        if not image_path:

            sample_images = list(

                (
                    ROOT
                    / "data"
                    / "sample_images"

                ).glob("*.png")

            )

            if not sample_images:

                raise FileNotFoundError(

                    "No PNG sample images found in "
                    "data/sample_images/"

                )

            image_path = str(
                sample_images[0]
            )


        # -------------------------------------------------
        # Call REAL Part 2 model
        # -------------------------------------------------

        image_path = resolve_image_path(image_path)

        result = classify_product_image(
            image_path
        )

        return {

            "tool_result": result

        }


    return {}


# =========================================================
# MOCK LLM RESPONSE
# =========================================================

def mock_llm_response(state: AgentState):

    intent = state["intent"]


    # =====================================================
    # BLOCKED INPUT
    # =====================================================

    if state.get("blocked"):

        return {

            "answer":
            (
                "I can't follow that instruction. "
                "Please ask a normal Flipkart support question."
            ),

            "source":
            "policy_kb",

            "confidence":
            1.0

        }


    # =====================================================
    # POLICY RESPONSE
    # =====================================================

    if intent == "policy":

        chunks = state.get(
            "retrieved_chunks",
            []
        )


        if not chunks:

            return {

                "answer":
                (
                    "I could not find a sufficiently "
                    "relevant policy in the knowledge base."
                ),

                "source":
                "policy_kb",

                "confidence":
                0.0

            }


        best_score = max(

            chunk["similarity"]

            for chunk in chunks

        )


        print(
            f"Best policy similarity: {best_score:.4f}"
        )

        print(
            f"Grounding threshold: "
            f"{GROUNDING_THRESHOLD:.4f}"
        )


        # -------------------------------------------------
        # OUTPUT-SIDE GROUNDEDNESS CHECK
        # -------------------------------------------------

        if best_score < GROUNDING_THRESHOLD:

            return {

                "answer":
                (
                    "I can't provide a policy answer because "
                    "the knowledge base did not contain "
                    "sufficiently relevant information."
                ),

                "source":
                "policy_kb",

                "confidence":
                round(
                    best_score,
                    4
                )

            }


        # -------------------------------------------------
        # Use strongest retrieved chunk
        # -------------------------------------------------

        best_chunk = max(

            chunks,

            key=lambda x:
            x["similarity"]

        )


        answer = (

            f"{best_chunk['text']} "

            f"(Policy document: "
            f"{best_chunk['doc_id']})"

        )


        return {

            "answer":
            answer,

            "source":
            "policy_kb",

            "confidence":
            round(
                best_score,
                4
            )

        }


    # =====================================================
    # RETURN RISK RESPONSE
    # =====================================================

    if intent == "return_risk":

        result = state.get(
            "tool_result",
            {}
        )


        # -------------------------------------------------
        # Missing context
        # -------------------------------------------------

        if "error" in result:

            return {

                "answer":
                (
                    "I don't have the order details needed "
                    "to calculate return risk in this fresh "
                    "conversation. Please provide the order "
                    "details so I can check it."
                ),

                "source":
                "return_risk_tool",

                "confidence":
                0.0

            }


        probability = result.get(
            "return_probability"
        )

        bucket = result.get(
            "risk_bucket"
        )


        if probability is None:

            return {

                "answer":
                "I could not calculate the return risk.",

                "source":
                "return_risk_tool",

                "confidence":
                0.0

            }


        answer = (

            f"The predicted return probability is "

            f"{probability:.2%}, "

            f"which places the order in the "

            f"{bucket} risk bucket. "

            f"The bucket is calibrated using "

            f"t*_rf = "

            f"{result.get('t_rf'):.4f}."

        )


        return {

            "answer":
            answer,

            "source":
            "return_risk_tool",

            "confidence":
            round(
                probability,
                4
            )

        }


    # =====================================================
    # PRODUCT CATEGORY RESPONSE
    # =====================================================

    if intent == "product_category":

        result = state.get(
            "tool_result",
            {}
        )


        category = result.get(
            "predicted_category"
        )

        confidence = result.get(
            "confidence",
            0.0
        )


        if not category:

            return {

                "answer":
                "I could not classify the product image.",

                "source":
                "image_classifier_tool",

                "confidence":
                0.0

            }


        answer = (

            f"The image is classified as "

            f"{category} "

            f"with "

            f"{confidence:.2%} confidence."

        )


        return {

            "answer":
            answer,

            "source":
            "image_classifier_tool",

            "confidence":
            round(
                confidence,
                4
            )

        }


    # =====================================================
    # UNKNOWN
    # =====================================================

    return {

        "answer":
        "I could not determine the request.",

        "source":
        "policy_kb",

        "confidence":
        0.0

    }


# =========================================================
# RESPONSE NODE
# =========================================================

def response_node(state: AgentState):

    response = mock_llm_response(
        state
    )


    return {

        "answer":
        response["answer"],

        "source":
        response["source"],

        "confidence":
        response["confidence"],

        # IMPORTANT:
        # This is the key that Conversation.ask()
        # will return.
        "response":
        response,

        "final_response":
        response

    }


# =========================================================
# CONDITIONAL ROUTING
# =========================================================

def route_after_intent(state: AgentState):

    if state.get("blocked"):

        return "response"


    intent = state["intent"]


    if intent == "policy":

        return "retrieval"


    if intent in [
        "return_risk",
        "product_category"
    ]:

        return "tool"


    return "response"


# =========================================================
# BUILD GRAPH
# =========================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )


    # -----------------------------------------------------
    # Four required nodes
    # -----------------------------------------------------

    graph.add_node(
        "intent",
        intent_node
    )

    graph.add_node(
        "retrieval",
        retrieve_policy_node
    )

    graph.add_node(
        "tool",
        tool_node
    )

    graph.add_node(
        "response",
        response_node
    )


    # -----------------------------------------------------
    # Entry point
    # -----------------------------------------------------

    graph.set_entry_point(
        "intent"
    )


    # -----------------------------------------------------
    # Conditional branching
    # -----------------------------------------------------

    graph.add_conditional_edges(

        "intent",

        route_after_intent,

        {

            "retrieval":
            "retrieval",

            "tool":
            "tool",

            "response":
            "response"

        }

    )


    # -----------------------------------------------------
    # Retrieval -> Response
    # -----------------------------------------------------

    graph.add_edge(
        "retrieval",
        "response"
    )


    # -----------------------------------------------------
    # Tool -> Response
    # -----------------------------------------------------

    graph.add_edge(
        "tool",
        "response"
    )


    # -----------------------------------------------------
    # Response -> END
    # -----------------------------------------------------

    graph.add_edge(
        "response",
        END
    )


    return graph.compile()


# =========================================================
# CREATE GRAPH
# =========================================================

GRAPH = build_graph()


# =========================================================
# CONVERSATION STATE
# =========================================================

class Conversation:

    """
    Maintains short-term conversational state.

    Same Conversation object:
        state is preserved.

    New Conversation object:
        state starts empty.
    """


    def __init__(
        self,
        conversation_id=None
    ):

        self.conversation_id = (
            conversation_id
        )


        self.state = {

            "messages": [],

            "order_id": None,

            "last_intent": None,

            "last_result": None

        }


    def ask(
        self,
        user_input,
        order_features=None
    ):

        """
        Send one user message through LangGraph.

        Parameters
        ----------
        user_input : str
            User's question.

        order_features : dict, optional
            Real order features used by Part 1 model.

        Returns
        -------
        dict
            Structured response containing:

            answer
            source
            confidence
        """


        # -------------------------------------------------
        # Store user message
        # -------------------------------------------------

        self.state["messages"].append(

            {

                "role":
                "user",

                "content":
                user_input

            }

        )


        # -------------------------------------------------
        # Invoke graph
        # -------------------------------------------------

        result = GRAPH.invoke(

            {

                "messages":
                self.state["messages"],

                "conversation_id":
                self.conversation_id,

                "order_id":
                self.state.get(
                    "order_id"
                ),

                "last_intent":
                self.state.get(
                    "last_intent"
                ),

                "last_result":
                self.state.get(
                    "last_result"
                ),

                "order_features":
                order_features,

                "user_input":
                user_input

            }

        )


        # -------------------------------------------------
        # Update order ID
        # -------------------------------------------------

        if result.get("order_id"):

            self.state["order_id"] = (
                result["order_id"]
            )


        # -------------------------------------------------
        # Update last intent
        # -------------------------------------------------

        if result.get("intent"):

            self.state["last_intent"] = (
                result["intent"]
            )


        # -------------------------------------------------
        # Update last tool result
        # -------------------------------------------------

        if result.get("tool_result"):

            self.state["last_result"] = (
                result["tool_result"]
            )


        # -------------------------------------------------
        # Get final response
        # -------------------------------------------------

        response = result.get(
            "response"
        )


        if response is None:

            response = {

                "answer":
                result.get(
                    "answer",
                    "No response generated."
                ),

                "source":
                result.get(
                    "source",
                    "policy_kb"
                ),

                "confidence":
                result.get(
                    "confidence",
                    0.0
                )

            }


        # -------------------------------------------------
        # Store assistant response
        # -------------------------------------------------

        self.state["messages"].append(

            {

                "role":
                "assistant",

                "content":
                response

            }

        )


        return response


    def get_state(self):

        """
        Return a copy of the current conversation state.
        """

        return self.state.copy()


    def reset(self):

        """
        Clear the conversation state.
        """

        self.state = {

            "messages": [],

            "order_id": None,

            "last_intent": None,

            "last_result": None

        }


# =========================================================
# MANUAL DEMO
# =========================================================

if __name__ == "__main__":

    print(
        "\n" +
        "=" * 70
    )

    print(
        "FLIPKART SUPPORT AGENT"
    )

    print(
        "=" * 70
    )


    conversation = Conversation(
        "demo-001"
    )


    response = conversation.ask(

        "What is the return window for footwear?"

    )


    print(

        json.dumps(

            response,

            indent=2

        )

    )