import re
import rdkit.Chem as Chem
from rdkit.Chem import rdMolTransforms


class ParameterExtractor:
    _pdb = None
    _out = None
    _atom_list = []
    _conformer = None
    p1 = None
    p2 = None
    r1 = None
    r2 = None
    bridge = None
    rh = None
    homo = None
    lumo = None
    dipole = None

    def __init__(self, filename: str):
        with open(f"./out/{filename}.out") as f:
            self._out = f.readlines()
        self._pdb = Chem.MolFromPDBFile(f"./pdb/{filename}.pdb", removeHs=False)
        self._conformer = self._pdb.GetConformer()
        self._atom_list = self._pdb.GetAtoms()
        self.__set_nbo_prop()
        self.p1, self.p2 = self._get_phosphorus()
        self.rh = next(atom for atom in self._atom_list if atom.GetSymbol() == "Rh")
        self.bridge = self.__get_bridge()
        self.homo, self.lumo = self.__get_homo_lumo()
        self.dipole = self.__get_dipole()

    def __extract_blocks_from_out(
            self,
            start_pattern: str,
            end_pattern: str,
            include_last_line=False
    ) -> list[list[str]]:
        blocks = []
        start_line = None
        end_line = None
        for i, line in enumerate(self._out):
            if re.search(start_pattern, line):
                start_line = i
            if re.search(end_pattern, line):
                end_line = i
            if start_line is not None and end_line is not None:
                blocks.append(self._out[start_line:end_line + include_last_line])
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

    def __get_dipole(self) -> float:
        dipole = [line for line in self._out if re.search(" X= ", line)][-1]
        if str.split(dipole)[-2] == "Tot=":
            return float(str.split(dipole)[-1])

    def __set_nbo_prop(self) -> None:
        nbo_blocks = self.__extract_nbo_blocks()
        for atom in self._atom_list:
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

    def _get_phosphorus(self) -> (Chem.Atom, Chem.Atom):
        p_list = list(filter(lambda x: x.GetSymbol() == "P", self._atom_list))
        return sorted(p_list, key=lambda x: float(x.GetProp("NBO")), reverse=True)

    def __get_bridge(self) -> Chem.Atom:
        p1_neighbors = set(map(lambda x: x.GetIdx(), self.p1.GetNeighbors()))
        p2_neighbors = set(map(lambda x: x.GetIdx(), self.p2.GetNeighbors()))
        bridge_idx = list(p1_neighbors.intersection(p2_neighbors))[0]
        return next(atom for atom in self._atom_list if atom.GetIdx() == bridge_idx)

    def getBondDistance(self, atom_1: Chem.Atom, atom_2: Chem.Atom) -> float:
        return rdMolTransforms.GetBondLength(self._conformer, atom_1.GetIdx(), atom_2.GetIdx())

    def getAngle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom) -> float:
        return rdMolTransforms.GetAngleDeg(self._conformer, atom_1.GetIdx(), atom_2.GetIdx(), atom_3.GetIdx())

    def getTorsionAngle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom, atom_4: Chem.Atom) -> float:
        return rdMolTransforms.GetDihedralDeg(self._conformer, atom_1.GetIdx(), atom_2.GetIdx(), atom_3.GetIdx(), atom_4.GetIdx())
