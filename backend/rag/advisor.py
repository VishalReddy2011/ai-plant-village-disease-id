import json
import logging
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Fallback structured advice in case of LLM API issues
MOCK_TREATMENT_PLANS = {
    "en": {
        "immediate_actions": [
            "Remove and discard severely infected leaves immediately - do not compost.",
            "Apply a preventative organic spray (e.g., copper-based fungicide or neem oil)."
        ],
        "this_week": [
            "Prune lower branches to improve airflow and reduce humidity around the base.",
            "Adjust irrigation timing to early morning to allow leaves to dry during the day."
        ],
        "prevention_measures": [
            "Implement a 3-year crop rotation sequence avoiding other Solanaceous crops.",
            "Clean all gardening tools and equipment after contact with affected plants."
        ]
    },
    "hi": {
        "immediate_actions": [
            "संक्रमित पत्तियों को तुरंत हटा दें और नष्ट कर दें - खाद न बनाएं।",
            "एक निवारक जैविक छिड़काव करें (जैसे, तांबा-आधारित कवकनाशी या नीम का तेल)।"
        ],
        "this_week": [
            "हवा के प्रवाह को बेहतर बनाने और निचले हिस्से में आर्द्रता कम करने के लिए निचली शाखाओं की छंटाई करें।",
            "सिंचाई का समय सुबह जल्दी तय करें ताकि दिन में पत्तियाँ सूख सकें।"
        ],
        "prevention_measures": [
            "अन्य सोलानेसी फसलों से बचते हुए 3 साल के फसल चक्र को लागू करें।",
            "संक्रमित पौधों के संपर्क में आने के बाद सभी बागवानी उपकरणों को साफ करें।"
        ]
    }
}

class TreatmentPlan(BaseModel):
    """Schema for the personalized treatment plan."""
    immediate_actions: list[str] = Field(description="Actionable steps to take today/immediately")
    this_week: list[str] = Field(description="Steps to complete during this week")
    prevention_measures: list[str] = Field(description="Long-term prevention steps and rotation recommendations")

class TreatmentAdvisor:
    """Manages Chroma vector queries and formats tailored LLM advisories."""
    def __init__(self, persist_dir: str, api_key: str, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
        
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Load ChromaDB using native PersistentClient
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="plant_handbook")
        
        # Initialize OpenAI Chat client
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.2,
            max_tokens=1024
        )

    def get_treatment_plan(
        self,
        crop: str,
        disease_name: str,
        local_conditions: dict,
        language: str = "en"
    ) -> dict:
        """Retrieves document context and uses ChatOpenAI to draft a tailored treatment plan."""
        crop_clean = crop.lower().strip()
        disease_clean = disease_name.lower().strip()
        
        # 1. Retrieve relevant disease details from vector DB
        query = f"Crop: {crop_clean}, Disease: {disease_clean}"
        context = "No direct matching handbook documents found."
        try:
            # Embed query using OpenAI embeddings
            query_vector = self.embeddings.embed_query(query)
            # Query collection using native client
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=2
            )
            if results and "documents" in results and results["documents"]:
                flat_docs = results["documents"][0]
                if flat_docs:
                    context = "\n\n".join(flat_docs)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            
        # 2. Construct the system/user instruction
        system_prompt = (
            "You are an expert, multilingual agricultural advisor specializing in regional Indian farming systems.\n"
            "Generate a highly specific, customized treatment plan based on the farmer's local conditions and crop science resources.\n"
            "Structure your response strictly as a JSON object matching this schema:\n"
            "{\n"
            "  \"immediate_actions\": [\"action 1\", \"action 2\"],\n"
            "  \"this_week\": [\"step 1\", \"step 2\"],\n"
            "  \"prevention_measures\": [\"measure 1\", \"measure 2\"]\n"
            "}\n"
            f"IMPORTANT: Write all plans and text fields in {language} (language code). Do not use English unless scientific names are required."
        )
        
        user_prompt = (
            f"--- CROP DATA ---\n"
            f"Crop Type: {crop}\n"
            f"Detected Condition: {disease_name}\n"
            f"\n"
            f"--- LOCAL CONDITIONS ---\n"
            f"Growth Stage: {local_conditions.get('growth_stage', 'Unknown')}\n"
            f"Severity of Spread: {local_conditions.get('severity', 'Mild')}\n"
            f"Irrigation Method: {local_conditions.get('irrigation', 'Not Specified')}\n"
            f"Access to Fungicides: {local_conditions.get('fungicide_access', 'Organic Only')}\n"
            f"Recent Weather: {local_conditions.get('weather', 'Normal')}\n"
            f"\n"
            f"--- HANDBOOK RESOURCE --- \n"
            f"{context}\n"
            f"\n"
            f"Construct the tailored list of Immediate actions, This week's actions, and Preventive measures."
        )

        # 3. Call ChatOpenAI with structured output
        try:
            structured_llm = self.llm.with_structured_output(TreatmentPlan)
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return {
                "immediate_actions": result.immediate_actions,
                "this_week": result.this_week,
                "prevention_measures": result.prevention_measures,
                "generated_by_llm": True
            }
        except Exception as e:
            logger.error(f"Structured plan generation failed, falling back: {e}")
            # Text prompt fallback in case structured output is not supported
            try:
                raw_response = self.llm.invoke([
                    {"role": "system", "content": system_prompt + "\nOutput raw JSON only."},
                    {"role": "user", "content": user_prompt}
                ])
                data = json.loads(raw_response.content)
                return {
                    "immediate_actions": data.get("immediate_actions", []),
                    "this_week": data.get("this_week", []),
                    "prevention_measures": data.get("prevention_measures", []),
                    "generated_by_llm": True
                }
            except Exception:
                # Return static translated mock data if everything else fails
                fallback_data = MOCK_TREATMENT_PLANS.get(language, MOCK_TREATMENT_PLANS["en"])
                return {
                    **fallback_data,
                    "generated_by_llm": False
                }
