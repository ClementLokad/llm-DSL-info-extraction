from typing import List, Dict, Any, Optional
from rag.core.base_chunker import CodeChunk
import agents.prepare_agent as prepare_agent
import csv
import os
from collections import deque

class ChunkSummarizer():
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.summary_agent = prepare_agent.prepare_summary_agent()
        self.summary_prompt = config.get('summary_prompt', "Summarize what the following code chunk does very briefly using keywords:")
        self.summary_list_path = config.get('summary_list_path', "")

    def generate_chunk_summary(self, chunk: CodeChunk) -> str:
        """generates a summary made by a LLM agent as a chunk metadata"""
        chunk_summary = self.summary_agent.generate_response(self.summary_prompt + "Chunk to summarize :" + chunk.content)

        #remove all line skips from the summary
        chunk_summary = chunk_summary.replace('\n', ' ').strip()

        return chunk_summary

    def get_last_index(self) -> int:
        """Reads only the last line of the file to get the last processed index (better alternative could be coded with seek)"""
        try:
            with open(self.summary_list_path, 'r', encoding='utf-8') as f:
                last_line_generator = deque(f, maxlen=1)
                
                # if file is empty
                if not last_line_generator:
                    return 0
                    
                last_line = last_line_generator[0]
                
                # keep index from last line
                last_index = last_line.split(',')[0]
                
                # add 1 to the last index to get the next index to process
                return int(last_index) + 1
                
        except FileNotFoundError:
            return 0   

    def generate_summary_file(self, chunks: List[CodeChunk], rebuild: bool = False) -> None:
        """Main function to generate a list of summaries using a list of chunks."""
        total_iterations = len(chunks)

        if rebuild:
            # start from scratch
            open(self.summary_list_path, 'w+').close()
            start_index = 0

        else:
            # recover the last processed index
            start_index = self.get_last_index()

        # print the current state
        print(f"--- Summary Task State ---")
        print(f"List length: {total_iterations}")
        print(f"Already done: {start_index}")
        print(f"To do: {total_iterations - start_index}")
        print("-" * 20)

        if not rebuild and start_index >= total_iterations:
            print("File is already complete. No more calculations needed.")

        else:
            with open(self.summary_list_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                try:
                    for i in range(start_index, total_iterations):
                        print(f"Treating chunk N°{i}...", end='\r')
                        
                        # generate summary
                        resultat = self.generate_chunk_summary(chunks[i])

                        # write summary to file
                        writer.writerow([i, resultat])

                        # write to file immediately to avoid data loss
                        f.flush() 
                        
                except KeyboardInterrupt:
                    print("\n\n🛑 Manual stop detected (CTRL+C) 🛑.")
                    print("Data calculated so far has been saved.")
                else:
                    print(f"\n Summaries generated successfully and saved '{self.summary_list_path}'.")
                return

    def get_summary_list(self) -> list:
        """Reads the summary file and returns a list of summaries"""
        results = []
        
        #if the file does not exist, return empty list
        if not os.path.exists(self.summary_list_path):
            return results, 0

        with open(self.summary_list_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                if row:
                    summary = row[1]  # get the summary from the second column
                    results.append(summary)
                    
        return results
    
    def get_summary_state(self, chunks: List[CodeChunk]) -> int:
        """Returns the number of summaries already generated"""
        return (f"{max(0,self.get_last_index()-1)}/{len(chunks)} summaries generated., ")