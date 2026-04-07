from typing import Dict, Any, List
from pipeline.agent_workflow.workflow_base import _tool_desc, Tool
from rag.core.base_retriever import RetrievalResult


class PriorEvidenceTool(Tool):
    """
    Tool to retrieve prior evidence/facts from accumulated evidence collected in previous tool calls.
    
    The accumulated_evidence is a dictionary where:
    - Keys: unique IDs as strings (UUIDs)
    - Values: RetrievalResult objects from previous tool calls
    """
    
    def retrieve_prior_evidence(
        self, 
        evidence_ids: List[str], 
        accumulated_evidence: Dict[str, RetrievalResult]
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve prior evidence items based on their IDs, tracking which IDs yield results.
        
        Args:
            evidence_ids: List of evidence IDs to retrieve (e.g., ["uuid1", "uuid2"])
            accumulated_evidence: Dictionary mapping IDs to RetrievalResult objects
            
        Returns:
            Dict mapping each requested evidence_id to its list of RetrievalResult objects.
            If an ID is not found, its list will be empty.
        """
        retrieved_results = {}
        
        for evidence_id in evidence_ids:
            if evidence_id in accumulated_evidence:
                result = accumulated_evidence[evidence_id]
                retrieved_results[evidence_id] = [result]  # List with single item
            else:
                # Explicitly mark missing IDs with empty list
                retrieved_results[evidence_id] = []
        
        return retrieved_results
    
    def format_results_by_source(
        self, 
        results_by_id: Dict[str, List[RetrievalResult]]
    ) -> tuple[Dict[str, List[Dict[str, Any]]], str]:
        """
        Format RetrievalResult objects grouped by source file, with explicit reporting of empty IDs.
        
        Args:
            results_by_id: Dict mapping evidence_ids to lists of RetrievalResult objects.
                          Empty lists indicate IDs that were not found.
            
        Returns:
            Tuple of (results_by_source dict, formatted_string)
        """
        results_by_source = {}
        empty_ids = []
        total_results = 0
        
        # Track which IDs have no results
        for evidence_id, results in results_by_id.items():
            if not results:
                empty_ids.append(evidence_id)
            else:
                for r in results:
                    total_results += 1
                    source = r.chunk.metadata.get('original_file_path', 'Unknown')
                    if source not in results_by_source:
                        results_by_source[source] = []
                    
                    line_start, line_end = r.chunk.get_line_range()
                    results_by_source[source].append({
                        "content": r.chunk.content,
                        "line_start": line_start,
                        "line_end": line_end
                    })
        
        # Format output grouped by source with line numbers
        raw_results_parts = []
        
        # First, show empty IDs explicitly
        if empty_ids:
            for empty_id in sorted(empty_ids):
                raw_results_parts.append(f"⚠️  Evidence ID '{empty_id}': No results found for requested ID")
        
        # Then show grouped results by source
        for source in sorted(results_by_source.keys()):
            source_results = results_by_source[source]
            if len(source_results) > 1:
                raw_results_parts.append(f"\n=== Source: {source} [{len(source_results)} results] ===")
            else:
                raw_results_parts.append(f"\n=== Source: {source} ===")
            
            for result in source_results:
                line_start = result["line_start"]
                line_end = result["line_end"]
                content = result["content"]
                raw_results_parts.append(f"[Lines {line_start}-{line_end}]:\n{content}")
        
        if not raw_results_parts:
            raw_results_str = "(No prior evidence found for any of the requested IDs.)"
        else:
            raw_results_str = "\n\n".join(raw_results_parts)
        
        return results_by_source, raw_results_str
    
    def get_description(self) -> Dict[str, Any]:
        """Return the Mistral-compatible tool schema for this tool."""
        return _tool_desc(
            name="prior_evidence_tool",
            description=(
                "Retrieve previously accumulated evidence from earlier retrieval operations. "
                "Use this to access facts that were discovered in earlier investigation steps "
                "without re-invoking retrieval tools."
            ),
            properties={
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of evidence IDs to retrieve (e.g., ['ev_uuid1', 'ev_uuid2']). "
                        "Each ID corresponds to a specific retrieval result from previous tool calls.\n"
                        "These IDs can be found in the VERIFIED FACTS section."
                    ),
                },
            },
            required=["evidence_ids"],
        )
        
