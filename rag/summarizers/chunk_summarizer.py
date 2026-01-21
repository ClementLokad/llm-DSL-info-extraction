from typing import List, Dict, Any, Optional
import json
import os
from rag.core.base_chunker import CodeChunk
import agents.prepare_agent as prepare_agent

class ChunkSummarizer():
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.summary_agent = prepare_agent.prepare_summary_agent()
        self.summary_prompt = config.get('summary_prompt', "Summarize what the following code chunk does briefly:")
        # Default to a .json extension
        self.summary_list_path = config.get('summary_list_path', "summaries.json")

    def generate_chunk_summary(self, chunk: CodeChunk) -> str:
        """Generates a summary using an LLM agent."""
        prompt = f"{self.summary_prompt}\n\nCODE:\n{chunk.content}\n\n### Summary:\n"
        chunk_summary = self.summary_agent.generate_response(prompt)
        return chunk_summary.strip()

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
        Updates the file after every chunk to prevent data loss.
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

        print(f"--- Summary Task State (JSON) ---")
        print(f"Total chunks: {total_iterations}")
        print(f"Already processed: {start_index}")
        print(f"To do: {max(0, total_iterations - start_index)}")
        print("-" * 20)

        if start_index >= total_iterations:
            print("Processing already complete.")
            return

        try:
            for i in range(start_index, total_iterations):
                print(f"Treating chunk N°{i}/{total_iterations-1}...", end='\r')
                
                summary = self.generate_chunk_summary(chunks[i])

                # Update local dictionary and save to file
                existing_data[str(i)] = summary
                self._save_json_data(existing_data)

        except KeyboardInterrupt:
            print("\n\n🛑 Manual stop detected (CTRL+C). Progress saved to JSON.")
        else:
            print(f"\n✅ All summaries saved to '{self.summary_list_path}'.")

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