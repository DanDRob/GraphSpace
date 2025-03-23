import json
import os
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple


class KnowledgeGraph:
    def __init__(self, data_path: str = "data/user_data.json"):
        """
        Initialize the knowledge graph.

        Args:
            data_path: Path to the JSON file containing user data.
        """
        self.data_path = data_path
        self.graph = nx.Graph()
        self.node_embeddings = {}
        self.data = self._load_data()
        self.build_graph()

    def _load_data(self) -> Dict:
        """
        Load data from the JSON file. If the file doesn't exist or is empty,
        return an empty data structure.

        Returns:
            Dict containing the loaded data.
        """
        if not os.path.exists(self.data_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            # Create empty data structure
            empty_data = {
                "notes": [],
                "tasks": [],
                "contacts": []
            }
            # Save the empty structure
            with open(self.data_path, 'w') as f:
                json.dump(empty_data, f)
            return empty_data

        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                # Ensure all required keys exist
                for key in ["notes", "tasks", "contacts"]:
                    if key not in data:
                        data[key] = []
                return data
        except json.JSONDecodeError:
            # Handle case where file exists but is empty or invalid
            return {"notes": [], "tasks": [], "contacts": []}

    def save_data(self):
        """Save the current data back to the JSON file."""
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def build_graph(self):
        """
        Build the knowledge graph from the loaded data.
        Create nodes for all entities and edges based on relationships.
        """
        # Clear existing graph
        self.graph.clear()

        # Add nodes for each entity type
        self._add_nodes_from_data()

        # Add edges based on relationships
        self._add_edges_for_notes()
        self._add_edges_for_tasks()
        self._add_edges_for_contacts()
        self._add_cross_entity_edges()

    def _add_nodes_from_data(self):
        """Add nodes to the graph for all entities in the data."""
        # Add notes
        for i, note in enumerate(self.data.get("notes", [])):
            node_id = f"note_{note.get('id', i)}"
            self.graph.add_node(
                node_id,
                type="note",
                data=note,
                title=note.get("title", ""),
                content=note.get("content", ""),
                tags=note.get("tags", []),
                created=note.get("created", ""),
                updated=note.get("updated", "")
            )

        # Add tasks
        for i, task in enumerate(self.data.get("tasks", [])):
            node_id = f"task_{task.get('id', i)}"
            self.graph.add_node(
                node_id,
                type="task",
                data=task,
                title=task.get("title", ""),
                description=task.get("description", ""),
                status=task.get("status", ""),
                due_date=task.get("due_date", ""),
                tags=task.get("tags", []),
                project=task.get("project", "")
            )

        # Add contacts
        for i, contact in enumerate(self.data.get("contacts", [])):
            node_id = f"contact_{contact.get('id', i)}"
            self.graph.add_node(
                node_id,
                type="contact",
                data=contact,
                name=contact.get("name", ""),
                email=contact.get("email", ""),
                phone=contact.get("phone", ""),
                organization=contact.get("organization", ""),
                tags=contact.get("tags", [])
            )

    def _add_edges_for_notes(self):
        """Add edges between notes based on shared tags and other relationships."""
        notes_nodes = [n for n, attr in self.graph.nodes(
            data=True) if attr["type"] == "note"]

        # Connect notes that share tags
        for i, node1 in enumerate(notes_nodes):
            for node2 in notes_nodes[i+1:]:
                node1_tags = set(self.graph.nodes[node1].get("tags", []))
                node2_tags = set(self.graph.nodes[node2].get("tags", []))

                # If they share tags, add an edge
                shared_tags = node1_tags.intersection(node2_tags)
                if shared_tags:
                    self.graph.add_edge(
                        node1, node2,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

    def _add_edges_for_tasks(self):
        """Add edges between tasks based on shared projects, tags, etc."""
        tasks_nodes = [n for n, attr in self.graph.nodes(
            data=True) if attr["type"] == "task"]

        # Connect tasks that share projects or tags
        for i, node1 in enumerate(tasks_nodes):
            for node2 in tasks_nodes[i+1:]:
                node1_data = self.graph.nodes[node1]
                node2_data = self.graph.nodes[node2]

                # Connect by project
                if (node1_data.get("project") and
                        node1_data.get("project") == node2_data.get("project")):
                    self.graph.add_edge(
                        node1, node2,
                        relationship="same_project",
                        project=node1_data.get("project"),
                        weight=1.0
                    )

                # Connect by shared tags
                node1_tags = set(node1_data.get("tags", []))
                node2_tags = set(node2_data.get("tags", []))
                shared_tags = node1_tags.intersection(node2_tags)

                if shared_tags:
                    self.graph.add_edge(
                        node1, node2,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

    def _add_edges_for_contacts(self):
        """Add edges between contacts based on shared organizations, tags, etc."""
        contacts_nodes = [n for n, attr in self.graph.nodes(
            data=True) if attr["type"] == "contact"]

        # Connect contacts in the same organization
        for i, node1 in enumerate(contacts_nodes):
            for node2 in contacts_nodes[i+1:]:
                node1_data = self.graph.nodes[node1]
                node2_data = self.graph.nodes[node2]

                # Connect by organization
                if (node1_data.get("organization") and
                        node1_data.get("organization") == node2_data.get("organization")):
                    self.graph.add_edge(
                        node1, node2,
                        relationship="same_organization",
                        organization=node1_data.get("organization"),
                        weight=1.0
                    )

                # Connect by shared tags
                node1_tags = set(node1_data.get("tags", []))
                node2_tags = set(node2_data.get("tags", []))
                shared_tags = node1_tags.intersection(node2_tags)

                if shared_tags:
                    self.graph.add_edge(
                        node1, node2,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

    def _add_cross_entity_edges(self):
        """Add edges between different entity types based on relationships."""
        # Connect notes to tasks by tags or mentions
        for note_node in [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "note"]:
            note_data = self.graph.nodes[note_node]
            note_tags = set(note_data.get("tags", []))
            note_content = note_data.get("content", "").lower()

            # Connect notes to tasks
            for task_node in [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "task"]:
                task_data = self.graph.nodes[task_node]
                task_tags = set(task_data.get("tags", []))

                # Connect by shared tags
                shared_tags = note_tags.intersection(task_tags)
                if shared_tags:
                    self.graph.add_edge(
                        note_node, task_node,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

                # Connect if task title is mentioned in note
                task_title = task_data.get("title", "").lower()
                if task_title and task_title in note_content:
                    self.graph.add_edge(
                        note_node, task_node,
                        relationship="mention",
                        weight=1.0
                    )

            # Connect notes to contacts
            for contact_node in [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "contact"]:
                contact_data = self.graph.nodes[contact_node]
                contact_name = contact_data.get("name", "").lower()

                # Connect if contact name is mentioned in note
                if contact_name and contact_name in note_content:
                    self.graph.add_edge(
                        note_node, contact_node,
                        relationship="mention",
                        weight=1.0
                    )

                # Connect by shared tags
                contact_tags = set(contact_data.get("tags", []))
                shared_tags = note_tags.intersection(contact_tags)
                if shared_tags:
                    self.graph.add_edge(
                        note_node, contact_node,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

        # Connect tasks to contacts
        for task_node in [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "task"]:
            task_data = self.graph.nodes[task_node]
            task_tags = set(task_data.get("tags", []))
            task_description = task_data.get("description", "").lower()

            for contact_node in [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "contact"]:
                contact_data = self.graph.nodes[contact_node]
                contact_name = contact_data.get("name", "").lower()

                # Connect if contact name is mentioned in task
                if contact_name and contact_name in task_description:
                    self.graph.add_edge(
                        task_node, contact_node,
                        relationship="mention",
                        weight=1.0
                    )

                # Connect by shared tags
                contact_tags = set(contact_data.get("tags", []))
                shared_tags = task_tags.intersection(contact_tags)
                if shared_tags:
                    self.graph.add_edge(
                        task_node, contact_node,
                        relationship="shared_tags",
                        shared_tags=list(shared_tags),
                        weight=len(shared_tags)
                    )

    def get_node_attributes(self, node_id: str) -> Dict:
        """Get all attributes for a given node."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return {}

    def get_connected_nodes(self, node_id: str, relationship: Optional[str] = None) -> List[str]:
        """
        Get all nodes connected to the given node, optionally filtered by relationship type.

        Args:
            node_id: ID of the node to get connections for
            relationship: Optional relationship type to filter by

        Returns:
            List of connected node IDs
        """
        if node_id not in self.graph:
            return []

        if relationship:
            connected = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data.get("relationship") == relationship:
                    connected.append(neighbor)
            return connected
        else:
            return list(self.graph.neighbors(node_id))

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all nodes of a specific type."""
        return [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == node_type]

    def search_nodes(self, query: str, node_type: Optional[str] = None) -> List[str]:
        """
        Search for nodes that match the query string.

        Args:
            query: Search query to match against node content
            node_type: Optional type to filter results by

        Returns:
            List of matching node IDs
        """
        query = query.lower()
        matching_nodes = []

        for node, attr in self.graph.nodes(data=True):
            # Filter by type if specified
            if node_type and attr.get("type") != node_type:
                continue

            # Search in different fields based on node type
            if attr.get("type") == "note":
                searchable_text = (
                    attr.get("title", "").lower() + " " +
                    attr.get("content", "").lower()
                )
                if query in searchable_text:
                    matching_nodes.append(node)

            elif attr.get("type") == "task":
                searchable_text = (
                    attr.get("title", "").lower() + " " +
                    attr.get("description", "").lower()
                )
                if query in searchable_text:
                    matching_nodes.append(node)

            elif attr.get("type") == "contact":
                searchable_text = (
                    attr.get("name", "").lower() + " " +
                    attr.get("email", "").lower() + " " +
                    attr.get("organization", "").lower()
                )
                if query in searchable_text:
                    matching_nodes.append(node)

        return matching_nodes

    def get_context_for_query(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant context for a user query.
        This is a simple search-based implementation, to be enhanced with GNN embeddings.

        Args:
            query: User query
            max_results: Maximum number of results to return

        Returns:
            List of context dictionaries with node information
        """
        # Simple keyword-based search as fallback
        matching_nodes = self.search_nodes(query)

        # Prepare context items
        context = []
        for node_id in matching_nodes[:max_results]:
            node_data = self.get_node_attributes(node_id)
            context_item = {
                "id": node_id,
                "type": node_data.get("type"),
                "content": ""
            }

            # Add specific content based on node type
            if node_data.get("type") == "note":
                context_item["content"] = node_data.get("content", "")
                context_item["title"] = node_data.get("title", "")
            elif node_data.get("type") == "task":
                context_item["content"] = node_data.get("description", "")
                context_item["title"] = node_data.get("title", "")
            elif node_data.get("type") == "contact":
                context_item["content"] = f"Name: {node_data.get('name', '')}, Email: {node_data.get('email', '')}, Organization: {node_data.get('organization', '')}"
                context_item["title"] = node_data.get("name", "")

            context.append(context_item)

        return context

    def add_note(self, note_data: Dict) -> str:
        """
        Add a new note to the graph.

        Args:
            note_data: Dictionary with note data

        Returns:
            ID of the new note node
        """
        # Generate ID if not provided
        if "id" not in note_data:
            note_data["id"] = str(len(self.data["notes"]) + 1)

        # Add to data
        self.data["notes"].append(note_data)

        # Add to graph
        node_id = f"note_{note_data['id']}"
        self.graph.add_node(
            node_id,
            type="note",
            data=note_data,
            title=note_data.get("title", ""),
            content=note_data.get("content", ""),
            tags=note_data.get("tags", []),
            created=note_data.get("created", ""),
            updated=note_data.get("updated", "")
        )

        # Rebuild edges for this note
        self._add_edges_for_single_note(node_id)

        # Save data
        self.save_data()

        return node_id

    def _add_edges_for_single_note(self, note_node: str):
        """Add edges for a single note node to other relevant nodes."""
        note_data = self.graph.nodes[note_node]
        note_tags = set(note_data.get("tags", []))
        note_content = note_data.get("content", "").lower()

        # Connect to other notes
        for other_node in self.get_nodes_by_type("note"):
            if other_node == note_node:
                continue

            other_tags = set(self.graph.nodes[other_node].get("tags", []))
            shared_tags = note_tags.intersection(other_tags)

            if shared_tags:
                self.graph.add_edge(
                    note_node, other_node,
                    relationship="shared_tags",
                    shared_tags=list(shared_tags),
                    weight=len(shared_tags)
                )

        # Connect to tasks
        for task_node in self.get_nodes_by_type("task"):
            task_data = self.graph.nodes[task_node]
            task_tags = set(task_data.get("tags", []))

            # Connect by shared tags
            shared_tags = note_tags.intersection(task_tags)
            if shared_tags:
                self.graph.add_edge(
                    note_node, task_node,
                    relationship="shared_tags",
                    shared_tags=list(shared_tags),
                    weight=len(shared_tags)
                )

            # Connect if task title is mentioned in note
            task_title = task_data.get("title", "").lower()
            if task_title and task_title in note_content:
                self.graph.add_edge(
                    note_node, task_node,
                    relationship="mention",
                    weight=1.0
                )

        # Connect to contacts
        for contact_node in self.get_nodes_by_type("contact"):
            contact_data = self.graph.nodes[contact_node]
            contact_name = contact_data.get("name", "").lower()

            # Connect if contact name is mentioned in note
            if contact_name and contact_name in note_content:
                self.graph.add_edge(
                    note_node, contact_node,
                    relationship="mention",
                    weight=1.0
                )

            # Connect by shared tags
            contact_tags = set(contact_data.get("tags", []))
            shared_tags = note_tags.intersection(contact_tags)
            if shared_tags:
                self.graph.add_edge(
                    note_node, contact_node,
                    relationship="shared_tags",
                    shared_tags=list(shared_tags),
                    weight=len(shared_tags)
                )

    def update_embeddings(self, node_embeddings: Dict[str, np.ndarray]):
        """
        Update the node embeddings from the GNN model.

        Args:
            node_embeddings: Dictionary mapping node IDs to embeddings
        """
        self.node_embeddings = node_embeddings

    def get_embedding_for_node(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Embedding array or None if not available
        """
        return self.node_embeddings.get(node_id)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all node embeddings."""
        return self.node_embeddings

    def to_networkx(self) -> nx.Graph:
        """Return the networkx graph representation."""
        return self.graph

    def to_pyg_data(self):
        """
        Convert the graph to a PyTorch Geometric data object.
        This method will be implemented when integrating with the GNN module.
        """
        # This will be implemented when integrating with PyTorch Geometric
        pass
