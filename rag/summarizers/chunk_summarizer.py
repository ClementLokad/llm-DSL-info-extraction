from typing import List, Dict, Any, Optional
import json
import os
from rag.core.base_chunker import CodeChunk
import agents.prepare_agent as prepare_agent

class ChunkSummarizer():
    """
    Chunk summarizer able to summarize chunks into concise summaries in order to create a json index of 
    summaries ready to be embedded.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.summary_agent = prepare_agent.prepare_summary_agent()
        self.summary_prompt = (
            "### System Prompt\n"
            "You are an expert software architect. Your task is to generate a dense,"
            "semantic summary of the provided code chunk to improve vector search retrieval.\n\n"
            "### Instructions\n"
            "Analyze the code and provide a concise summary that includes:\n"
            "1. **Purpose:** What is the primary goal of this code? (e.g., 'Forecast next month's sales')\n"
            "2. **Key Definitions:** Mention what is computed or shown.\n"
            "3. **Key Dependencies:** Mention which resources are read or modified.\n"
            "4. **Logic Flow:** Summarize the core algorithm or transformation occurring.\n\n"
            "### Constraint\n"
            "- Do not use conversational filler (e.g., 'This code is about...').\n"
            "- Use technical keywords that a developer might use when searching.\n"
            "- Keep the summary between 50-150 words to maintain embedding density.\n\n"
            "### Code Chunk to summarize\n"
        )
        # Default to a .json extension
        self.summary_list_path = config.get('summary_list_path', "summaries.json")

    def generate_chunk_summary(self, chunk: CodeChunk) -> str:
        """Generates a summary using an LLM agent."""
        prompt = f"{self.summary_prompt}\n\nCODE:\n{chunk.content}\n\n### Summary\n"
        chunk_summary = self.summary_agent.generate_response(prompt)
        chunk_summary = f"From script: {chunk.metadata.get('original_file_path', 'Unknown Source')}\n" + chunk_summary.strip()
        return chunk_summary

    def _load_json_data(self) -> Dict[str, str]:
        """Helper to load existing summaries from the JSON file."""
        if not os.path.exists(self.summary_list_path) or os.path.getsize(self.summary_list_path) == 0:
            return {}
        try:
            with open(self.summary_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_json_data(self, data: Dict[str, str]):
        """Helper to save the entire dictionary to JSON."""
        with open(self.summary_list_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def generate_summary_file(self, chunks: List[CodeChunk], rebuild: bool = False) -> None:
        """
        Processes chunks and saves summaries to a JSON file.
        Saves data only when exiting the loop or on KeyboardInterrupt for efficiency.
        """
        total_iterations = len(chunks)
        
        # Load existing data or start fresh
        existing_data = {} if rebuild else self._load_json_data()
        
        # Determine where to start based on the highest integer key in the JSON
        if not existing_data:
            start_index = 0
        else:
            # Keys in JSON are strings, convert to int to find the max
            processed_indices = [int(k) for k in existing_data.keys()]
            start_index = max(processed_indices) + 1

        if start_index >= total_iterations:
            print(f"--- Summary Task State (JSON) ---")
            print(f"Processing already complete. {total_iterations}/{total_iterations} Summaries already generated")
            return

        try:
            print(f"--- Summary Task State (JSON) ---")
            print(f"Total chunks: {total_iterations}")

            for i in range(start_index, total_iterations):
                print(f"Treating chunk N°{i}/{total_iterations-1}...", end='\r')

                summary = self.generate_chunk_summary(chunks[i])

                # Update local dictionary (don't save yet)
                existing_data[str(i)] = summary

        except KeyboardInterrupt:
            print("\n\n🛑 Manual stop detected (CTRL+C). Saving progress...")
            self._save_json_data(existing_data)
            print("Progress saved to JSON.")
        except Exception as e:
            print(f"The following error occured: {e}")
            self._save_json_data(existing_data)
            print("Progress saved to JSON.")
        else:
            print(f"\n✅ Saving all summaries to '{self.summary_list_path}'...")
            self._save_json_data(existing_data)

    def get_summary_list(self) -> List[str]:
        """Returns a list of summary strings ordered by index."""
        data = self._load_json_data()
        # Sort by key to ensure order [0, 1, 2...]
        sorted_keys = sorted(data.keys(), key=int)
        return [data[k] for k in sorted_keys]

    def get_summary_state(self, chunks: List[CodeChunk]) -> str:
        """Returns a progress string."""
        data = self._load_json_data()
        count = len(data)
        return f"{count}/{len(chunks)} summaries generated."