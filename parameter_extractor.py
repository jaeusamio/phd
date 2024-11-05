import re
import numpy as np
import rdkit.Chem as Chem
from openbabel import pybel
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDetermineBonds


class ParameterExtractor:
    __out = None
    __atom_list = []
    __conformer: Chem.Conformer = None
    mol: Chem.Mol = None
    p1: Chem.Atom = None
    p2: Chem.Atom = None
    r1: Chem.Atom = None
    r2: Chem.Atom = None
    r3: Chem.Atom = None
    r4: Chem.Atom = None
    bridge: Chem.Atom = None
    rh: Chem.Atom = None
    cod: list[Chem.Atom] = None
    c1: Chem.Atom = None
    c2: Chem.Atom = None
    c3: Chem.Atom = None
    c4: Chem.Atom = None
    homo: float = None
    lumo: float = None
    dipole: float = None

    def __init__(self, filename: str):
        with open(f"./out/spe/{filename}.out") as f:
            self.__out = f.readlines()
        mol = next(pybel.readfile("out", f"./out/spe/{filename}.out"))
        xyz = mol.write("xyz")
        self.mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(self.mol)
        self.__conformer = self.mol.GetConformer()
        self.__atom_list = self.mol.GetAtoms()
        self.__set_nbo_prop()
        self.p1, self.p2 = self.__get_phosphorus()
        self.rh = next(atom for atom in self.__atom_list if atom.GetSymbol() == "Rh")
        self.bridge = self.__get_bridge()
        self.r1, self.r2 = self.__get_p_substituents(self.p1)
        self.r3, self.r4 = self.__get_p_substituents(self.p2)
        self.homo, self.lumo = self.__get_homo_lumo()
        self.dipole = self.__get_dipole()
        self.cod = self.__get_cod()
        self.c1, self.c2, self.c3, self.c4 = self.__get_c_substituents()

    def __extract_blocks_from_out(
            self,
            start_pattern: str,
            end_pattern: str,
            include_last_line=False
    ) -> list[list[str]]:
        blocks = []
        start_line = None
        end_line = None
        for i, line in enumerate(self.__out):
            if re.search(start_pattern, line):
                start_line = i
            if re.search(end_pattern, line):
                end_line = i
            if start_line is not None and end_line is not None:
                blocks.append(self.__out[start_line:end_line + include_last_line])
                start_line = None
                end_line = None
        return blocks

    def __extract_nbo_blocks(self) -> list[list[str]]:
        start_pattern = r"Summary of Natural Population Analysis\:"
        end_pattern = r"\* Total \*"
        nbo_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, include_last_line=True)
        return nbo_blocks

    def __extract_orbital_blocks(self) -> list[list[str]]:
        start_pattern = r"Orbital symmetries"
        end_pattern = r"Condensed to atoms \(all electrons\)"
        orbital_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern)
        return orbital_blocks

    def __extract_bond_occupancy_blocks(self) -> list[list[str]]:
        start_pattern = r"\(Occupancy\)   Bond orbital\/ Coefficients\/ Hybrids"
        end_pattern = r"\n\n\n"
        bond_occupancy_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern)
        return bond_occupancy_blocks

    def __get_dipole(self) -> float:
        dipole = [line for line in self.__out if re.search(" X= ", line)][-1]
        if str.split(dipole)[-2] == "Tot=":
            return float(str.split(dipole)[-1])

    def __set_nbo_prop(self) -> None:
        nbo_blocks = self.__extract_nbo_blocks()
        for atom in self.__atom_list:
            _atom_idx = str(atom.GetIdx() + 1)
            _atom_symbol = atom.GetSymbol()
            for line in nbo_blocks[-1]:
                pattern = fr"{_atom_symbol}\s+{_atom_idx}.+?\d"
                if re.search(pattern, line):
                    _atom_nbo = str.split(line)[2]
                    atom.SetProp("NBO", _atom_nbo)
                    break

    def __get_homo_lumo(self) -> (float, float):
        orbital_blocks = self.__extract_orbital_blocks()
        homo = None
        for line in orbital_blocks[-1][::-1]:
            pattern = r"Alpha\s+occ"
            if re.search(pattern, line):
                homo = str.split(line)[-1]
                break
        lumo = None
        for line in orbital_blocks[-1]:
            pattern = r"Alpha\s+virt\."
            if re.search(pattern, line):
                lumo = str.split(line)[4]
                break
        return float(homo), float(lumo)

    def __get_phosphorus(self) -> (Chem.Atom, Chem.Atom):
        p_list = list(filter(lambda x: x.GetSymbol() == "P", self.__atom_list))
        return sorted(p_list, key=lambda x: float(x.GetProp("NBO")), reverse=True)

    def __get_bridge(self) -> Chem.Atom:
        p1_neighbors = set(map(lambda x: x.GetIdx(), self.p1.GetNeighbors()))
        p2_neighbors = set(map(lambda x: x.GetIdx(), self.p2.GetNeighbors()))
        bridge_idx = list(p1_neighbors.intersection(p2_neighbors))[0]
        return next(atom for atom in self.__atom_list if atom.GetIdx() == bridge_idx)

    def get_bond_distance(self, atom_1: Chem.Atom, atom_2: Chem.Atom) -> float:
        return rdMolTransforms.GetBondLength(self.__conformer, atom_1.GetIdx(), atom_2.GetIdx())

    def get_angle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom) -> float:
        return rdMolTransforms.GetAngleDeg(self.__conformer, atom_1.GetIdx(), atom_2.GetIdx(), atom_3.GetIdx())

    def get_torsion_angle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom, atom_4: Chem.Atom) -> float:
        return rdMolTransforms.GetDihedralDeg(
            self.__conformer,
            atom_1.GetIdx(),
            atom_2.GetIdx(),
            atom_3.GetIdx(), atom_4.GetIdx()
        )

    def distance_to_plane(self, atom: Chem.Atom) -> float:
        # Get the coordinates of the three atoms defining the plane
        coords = self.mol.GetConformer().GetPositions()

        p1 = coords[self.p1.GetIdx()]
        p2 = coords[self.p2.GetIdx()]
        rh = coords[self.rh.GetIdx()]

        # Calculate the normal vector of the plane
        v1 = p2 - p1
        v2 = rh - p1
        normal = np.cross(v1, v2)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Get the coordinate of the atom to evaluate
        p_atom = coords[atom.GetIdx()]

        # Calculate the point-plane distance
        d = np.dot(normal, p_atom - p1)

        return d

    def __get_p_substituents(self, p: Chem.Atom) -> (Chem.Atom, Chem.Atom):
        substituents = list(filter(
            lambda x: x.GetIdx() != self.bridge.GetIdx() and x.GetIdx() != self.rh.GetIdx(),
            p.GetNeighbors())
        )
        r1, r2 = sorted(substituents, key=lambda x: self.distance_to_plane(x))
        return r1, r2

    def __get_cod(self) -> list[Chem.Atom]:
        cod = Chem.MolFromSmiles("C1CC=CCCC=C1")
        cyclooctane = Chem.MolFromSmiles("C1CCCCCCC1")  # Pdb conversions shows structure of COD as single bonds

        cod_carbons = self.mol.GetSubstructMatch(cod)
        if len(cod_carbons) == 0:
            cod_carbons = self.mol.GetSubstructMatch(cyclooctane)
        cod_carbons = [self.mol.GetAtomWithIdx(c) for c in cod_carbons]
        cod_atoms = cod_carbons
        for c in cod_carbons:
            cod_hydrogens = [h for h in c.GetNeighbors() if h.GetSymbol() == "H"]
            cod_atoms.extend(cod_hydrogens)

        return cod_atoms

    def __get_c_substituents(self) -> (Chem.Atom, Chem.Atom, Chem.Atom, Chem.Atom):
        cod_rh = [a for a in self.rh.GetNeighbors() if a.GetSymbol() == "C"]
        cod_c_info = [
            (c, self.get_bond_distance(self.p1, c), self.distance_to_plane(c), self.get_angle(self.p1, self.rh, c)) for
            c in cod_rh]
        angle_sorted = sorted(cod_c_info, key=lambda x: x[3], reverse=True)
        plane_distance_sorted = sorted(angle_sorted[:2], key=lambda x: x[2], reverse=True)
        c1, c2 = list(map(lambda x: x[0], plane_distance_sorted))
        plane_distance_sorted = sorted(angle_sorted[2:], key=lambda x: x[2], reverse=True)
        c3, c4 = list(map(lambda x: x[0], plane_distance_sorted))
        return c1, c2, c3, c4


class AtomOnPlaneException(Exception):
    pass


if __name__ == "__main__":
    comp = ParameterExtractor("l_2_SPE")
