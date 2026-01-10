
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from utils.helpers import normalize_text
from azure.ai.inference.models import ToolMessage, AssistantMessage 
import os
import logging
logger = logging.getLogger(__name__)

class AzureOpenAI:
    def __init__(self):

        endpoint = os.getenv("AZURE_OPENAI_INFERENCE_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.model = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        self.system_message = "Your are a helpful AI assistant that helps people find information."

        logger.info("AzureOpenAI client initialized.")

    def fetch_document(self):
        """
        Fetch all the text from all the PDF documents in the "documents" folder.

        Returns:
            str: The concatenated text from all the PDF documents.
        """
        try:
            from pypdf import PdfReader
            documents_folder = "documents"
            pdf_files = [f for f in os.listdir(f"{documents_folder}") if f.endswith(".pdf")]

            clean_doc_text = ""
            for pdf_file in pdf_files:
                pdf_path = os.path.join(f"{documents_folder}", pdf_file)
                reader = PdfReader(pdf_path)

                raw_text = ""
                for page in reader.pages:
                    raw_text += page.extract_text() + " "

                clean_doc_text += normalize_text(raw_text)

            return clean_doc_text.strip()
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            raise e
    
    def create_tool_msg(self, tool_data : dict):
        """
        Create a list of ToolMessage objects from the given tool data.

        Args:
            tool_data (dict): A dictionary containing tool IDs as keys and the corresponding tool outputs as values.

        Returns:
            list[ToolMessage]: A list of ToolMessage objects representing the tool calls.
        """

        try:
            tool_messages = []
            for tool_id, tool_output in tool_data.items():
                # Create a ToolMessage for each tool call (Azure OPENAI specific)
                tool_messages.append(
                    ToolMessage(
                        tool_call_id=tool_id,
                        content=tool_output
                    )
                )
            return tool_messages
        except Exception as e:
            logger.error(f"Error creating tool messages: {e}")
            raise e
    
    def build_context(self,vector_results):
        """
        Build the context string from the given vector results.

        The context string is built by concatenating the source and text of each vector result,
        separated by a line break and a separator string ("----------").

        Args:
            vector_results (list): A list of dictionaries containing the vector results.

        Returns:
            str: The context string built from the vector results.
        """

        context_chunks = []
        try:
            for i, r in enumerate(vector_results):
                source = r["metadata"].get("source", "Unknown")
                text = r["metadata"]["text"].strip()

                context_chunks.append(
                    f"[Source: {source}]\n{text}"
                )

            return "\n\n----------\n\n".join(context_chunks)
        except Exception as e:
            logger.error(f"Error building context: {e}")
            raise e

    def send_tool_msg(self, original_prompt: str, tool_messages : list, tool_calls):
        
        """
        Send the tool messages to the follow-up ask model and return the final response.

        Args:
            original_prompt (str): The original prompt from the user.
            tool_messages (list): A list of tool messages to send to the follow-up ask model.
            tool_calls (dict): A dictionary containing the tool calls to append to the assistant message.

        Returns:
            str: The final response from the follow-up ask model.
        """
        try:
            messages=[
                        SystemMessage(content=self.system_message), 
                        UserMessage(content=original_prompt)
                    ]
            # Append assistant tool call as message
            messages.append(
                AssistantMessage(
                    tool_calls=tool_calls,
                    content=None
                )
            )
            messages.extend(tool_messages)
            logger.info("Sending tool messages to follow-up ask model.")
            # Follow-up ask model to respond with results
            followup_response = self.model.complete(
                messages=messages,
                model=self.deployment_name,
                temperature=0.7,
                top_p=1.0,
            )
            final_message = followup_response.choices[0].message.content

            return final_message.strip()
        except Exception as e:
            logger.error(f"Error sending tool messages: {e}")
            raise e

    def generate_response_v1(self, user_query: str, raw_context: str, memory : list) -> str:

        try:
            logger.info("Generating response with context retrieved. v1 method called.")
            processed_context = self.build_context(raw_context)
            response = self.model.complete(
                messages = [
                    SystemMessage(content=self.system_message),
                    SystemMessage(content=f"""You are provided with the following retrieved context. Use ONLY this context to answer the question. CONTEXT: {processed_context}"""),
                ] + memory +
                [UserMessage(content=user_query)],
                model=self.deployment_name,
                temperature=0.7,
                top_p=1.0
                
            )

            return response
        except Exception as e:
            logger.error(f"Error generating response generate_response_v1: {e}")
            raise e
        

    def generate_response_v0(self, user_query: str, memory : list) -> str:
        try:
            logger.info("Generating response without context. v0 method called.")
            response = self.model.complete(
                messages = [
                    SystemMessage(content=self.system_message)
                ] + memory +
                [UserMessage(content=user_query)],
                model=self.deployment_name,
                temperature=0.7,
                top_p=1.0,
                tool_choice="auto",
                tools=[
                    {
                        "id": "fetch document",
                        "type": "function",
                        "function": {
                            "name": "fetch_document",
                            "description": "A tool to fetch all relevant document information about AcmeCloud.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                },
                                "required": []
                            
                            }
                        }
                    }
                ],
                
            )

            try:
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    tool_data = {}
                    for tool_call in tool_calls:
                        print(tool_call)
                        if tool_call.function["name"] == "fetch_document":
                            tool_output = self.fetch_document()
                            tool_data[tool_call["id"]] = tool_output

                    tool_messages = self.create_tool_msg(tool_data)
                    final_response = self.send_tool_msg(user_query, tool_messages, tool_calls)
                    return final_response
                else:
                    ai_response =  response.choices[0].message.content.strip()
                    return ai_response
            except Exception as e:
                logger.error(f"Error processing tool calls in generate_response_v0: {e} ")
                raise e
        except Exception as e:
            logger.error(f"Error generating response v0: {e}")
            raise e
