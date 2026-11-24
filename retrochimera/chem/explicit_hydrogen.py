"""Materialize template-bound hydrogens without changing encoder atom indices."""

from typing import Iterator, Optional

from rdkit import Chem


def hydrogen_inputs(
    mol: Chem.Mol, query: Chem.Mol
) -> Iterator[tuple[Chem.Mol, list[Optional[int]]]]:
    size = mol.GetNumAtoms()
    expanded = Chem.AddHs(mol)
    for atom in expanded.GetAtoms():
        if atom.GetIdx() >= size:
            atom.SetHybridization(Chem.HybridizationType.S)
    parameters = Chem.RemoveHsParameters()
    parameters.removeMapped = False
    # Full hydrogenation changes degree and H-count predicates; check those after pruning.
    relaxed_query = Chem.RWMol(query)
    for atom in query.GetAtoms():
        relaxed_query.ReplaceAtom(
            atom.GetIdx(),
            Chem.AtomFromSmarts(f"[#{atom.GetAtomicNum()}]" if atom.GetAtomicNum() else "*"),
        )
    seen = set()
    for match in expanded.GetSubstructMatches(relaxed_query, maxMatches=1000):
        # Deduplicate equivalent H choices within RDKit's bounded match set.
        sites = tuple(
            sorted(
                expanded.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in match if i >= size
            )
        )
        if sites in seen:
            continue
        seen.add(sites)
        graph = Chem.Mol(expanded)
        for atom in graph.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atom.SetAtomMapNum(int(atom.GetIdx() in match or atom.GetIdx() < size))
        # Unbound H must remain implicit so rewriting can change its heavy atom's valence.
        graph = Chem.RemoveHs(graph, parameters)
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        if not graph.HasSubstructMatch(query):
            continue
        # Only appended H atoms were removed, so original input indices are unchanged.
        yield graph, list(range(size)) + [None] * (graph.GetNumAtoms() - size)
