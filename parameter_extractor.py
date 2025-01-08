import re
import numpy as np
from enum import Enum
import rdkit.Chem as Chem
from openbabel import pybel
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDetermineBonds
from morfeus.buried_volume import BuriedVolume


class ParameterExtractor:
    __out: list[str]
    __out_no_rh: list[str]
    __conformer: Chem.Conformer = None
    __verbose: bool
    mol: Chem.Mol
    p1: Chem.Atom
    p2: Chem.Atom
    r1: Chem.Atom
    r2: Chem.Atom
    r3: Chem.Atom
    r4: Chem.Atom
    bridge: Chem.Atom
    rh: Chem.Atom
    cod: list[Chem.Atom]
    c1: Chem.Atom
    c2: Chem.Atom
    c3: Chem.Atom
    c4: Chem.Atom
    homo: float
    homo_no_rh: float
    lumo: float
    lumo_no_rh: float
    dipole: float
    dipole_no_rh: float

    def __init__(self, path: str, verbose: bool = False, no_rh_path: str = None, include_spe_prop: bool = True):
        self.__verbose = verbose
        include_no_rh = no_rh_path is not None
        with open(path) as f:
            self.__out = f.readlines()
        if include_no_rh:
            with open(no_rh_path) as f:
                self.__out_no_rh = f.readlines()
        mol = next(pybel.readfile("out", path))
        xyz = mol.write("xyz")
        self.mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(self.mol)
        self.__conformer = self.mol.GetConformer()
        self.__set_charge_prop(no_rh=False)
        if include_spe_prop:
            self.__set_bond_prop(no_rh=False)
            self.__set_antibond_prop(no_rh=False)
            self.__set_nmr_property(no_rh=False)
            self.__set_lone_pair_prop(no_rh=False)
            self.homo, self.lumo = self.__get_homo_lumo(no_rh=False)
            self.dipole = self.__get_dipole(no_rh=False)
        self.rh = next(atom for atom in self.mol.GetAtoms() if atom.GetSymbol() == "Rh")
        self.cod = self.__get_cod()
        self.p1, self.p2 = self.__get_phosphorus()
        self.bridge = self.__get_bridge()
        self.r1, self.r2 = self.__get_p_substituents(self.p1)
        self.r3, self.r4 = self.__get_p_substituents(self.p2)
        self.c1, self.c2, self.c3, self.c4 = self.__get_rh_c_substituents()
        if include_no_rh:
            self.__set_no_rh_numbering()
            self.__set_charge_prop(no_rh=True)
            self.__set_bond_prop(no_rh=True)
            self.__set_antibond_prop(no_rh=True)
            self.__set_nmr_property(no_rh=True)
            self.__set_lone_pair_prop(no_rh=True)
            self.__set_lone_pair_orbital_distribution()
            self.homo_no_rh, self.lumo_no_rh = self.__get_homo_lumo(no_rh=True)
            self.dipole_no_rh = self.__get_dipole(no_rh=True)

    def __extract_blocks_from_out(
            self,
            start_pattern: str,
            end_pattern: str,
            no_rh: bool,
            include_last_line: bool = False
    ) -> list[list[str]]:
        blocks = []
        start_line = None
        end_line = None
        if no_rh:
            out = self.__out_no_rh
        else:
            out = self.__out
        for i, line in enumerate(out):
            if re.search(start_pattern, line):
                start_line = i
            if re.search(end_pattern, line):
                end_line = i
            if start_line is not None and end_line is not None:
                blocks.append(out[start_line:end_line + include_last_line])
                start_line = None
                end_line = None
        return blocks

    def __extract_nbo_charge_blocks(self, no_rh: bool) -> list[list[str]]:
        start_pattern = r"Summary of Natural Population Analysis\:"
        end_pattern = r"\* Total \*"
        nbo_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, no_rh, include_last_line=True)
        return nbo_blocks

    def __extract_orbital_blocks(self, no_rh: bool) -> list[list[str]]:
        start_pattern = r"Orbital symmetries"
        end_pattern = r"Condensed to atoms \(all electrons\)"
        orbital_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, no_rh)
        return orbital_blocks

    def __extract_nmr_blocks(self, no_rh: bool) -> list[list[str]]:
        start_pattern = r"SCF GIAO Magnetic shielding tensor \(ppm\)\:"
        end_pattern = r"End of Minotr"
        nmr_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, no_rh)
        return nmr_blocks

    def __extract_bond_occupancy_blocks(self, no_rh: bool) -> list[list[str]]:
        start_pattern = r"Natural Bond Orbitals \(Summary\)\:"
        end_pattern = r"NATURAL LOCALIZED MOLECULAR ORBITAL \(NLMO\) ANALYSIS"
        bond_occupancy_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, no_rh)
        return bond_occupancy_blocks

    def __extract_orbital_occupancy_blocks(self, no_rh: bool) -> list[list[str]]:
        start_pattern = r"\(Occupancy\)   Bond orbital\/ Coefficients\/ Hybrids"
        end_pattern = r"NHO Directionality and \"Bond Bending\" \(deviations from line of nuclear centers\)"
        bond_occupancy_blocks = self.__extract_blocks_from_out(start_pattern, end_pattern, no_rh)
        return bond_occupancy_blocks

    def __get_dipole(self, no_rh: bool) -> float:
        if no_rh:
            out = self.__out_no_rh
        else:
            out = self.__out
        dipole = [line for line in out if re.search(" X= ", line)][-1]
        if str.split(dipole)[-2] == "Tot=":
            return float(str.split(dipole)[-1])

    def __set_charge_prop(self, no_rh: bool) -> None:
        charge_blocks = self.__extract_nbo_charge_blocks(no_rh)
        prop = Property.CHARGE_NO_RH.value if no_rh else Property.CHARGE.value
        for atom in self.mol.GetAtoms():
            _atom_idx = atom.GetAtomMapNum() if no_rh else str(atom.GetIdx() + 1)
            _atom_symbol = atom.GetSymbol()
            for line in charge_blocks[-1]:
                pattern = fr"{_atom_symbol}\s+{_atom_idx}.+?\d"
                if re.search(pattern, line):
                    _atom_nbo = str.split(line)[2]
                    atom.SetProp(prop, _atom_nbo)
                    break

    def __set_nmr_property(self, no_rh: bool) -> None:
        nmr_blocks = self.__extract_nmr_blocks(no_rh)
        prop_isotropic = Property.NMR_ISOTROPIC_NO_RH.value if no_rh else Property.NMR_ISOTROPIC.value
        prop_anisotropic = Property.NMR_ANISOTROPIC_NO_RH.value if no_rh else Property.NMR_ANISOTROPIC.value

        for atom in self.mol.GetAtoms():
            atom_idx = atom.GetAtomMapNum() if no_rh else str(atom.GetIdx() + 1)
            atom_symbol = atom.GetSymbol()
            for line in nmr_blocks[-1]:
                pattern = fr"{atom_idx}\s+{atom_symbol}\s+Isotropic"
                if re.search(pattern, line):
                    _atom_nmr_isotropic = line.split()[-4]
                    _atom_nmr_anisotropic = line.split()[-1]
                    atom.SetProp(prop_isotropic, _atom_nmr_isotropic)
                    atom.SetProp(prop_anisotropic, _atom_nmr_anisotropic)

    def __set_bond_prop(self, no_rh: bool) -> None:
        bond_occupancy_blocks = self.__extract_bond_occupancy_blocks(no_rh)
        prop_occ = Property.BOND_OCCUPANCY_NO_RH.value if no_rh else Property.BOND_OCCUPANCY.value
        prop_eng = Property.BOND_ENERGY_NO_RH.value if no_rh else Property.BOND_ENERGY.value

        for bond in self.mol.GetBonds():
            found = None
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            atom_symbol_begin = begin_atom.GetSymbol()
            atom_idx_begin = begin_atom.GetAtomMapNum() if no_rh else begin_atom.GetIdx() + 1
            atom_symbol_end = end_atom.GetSymbol()
            atom_idx_end = end_atom.GetAtomMapNum() if no_rh else end_atom.GetIdx() + 1

            pattern_1 = fr"BD\s+\(\s+1\)\s*{atom_symbol_begin}\s+{atom_idx_begin}\s+\-\s*{atom_symbol_end}\s+{atom_idx_end}"
            pattern_2 = fr"BD\s+\(\s+1\)\s*{atom_symbol_end}\s+{atom_idx_end}\s+\-\s*{atom_symbol_begin}\s+{atom_idx_begin}"

            for line in bond_occupancy_blocks[-1]:
                if re.search(pattern_1, line):
                    bond_occupancy = str.split(line)[-3]
                    bond_energy = str.split(line)[-2]
                    bond.SetProp(prop_occ, bond_occupancy)
                    bond.SetProp(prop_eng, bond_energy)
                    found = True
                    break
                elif re.search(pattern_2, line):
                    bond_occupancy = str.split(line)[-3]
                    bond_energy = str.split(line)[-2]
                    bond.SetProp(prop_occ, bond_occupancy)
                    bond.SetProp(prop_eng, bond_energy)
                    found = True
                    break
            if not found and self.__verbose:
                print(atom_symbol_begin, atom_idx_begin, "-", atom_symbol_end, atom_idx_end, ": Bond not found")

    def __set_antibond_prop(self, no_rh: bool) -> None:
        bond_occupancy_blocks = self.__extract_bond_occupancy_blocks(no_rh)
        prop_occ = Property.ANTIBOND_OCCUPANCY_NO_RH.value if no_rh else Property.ANTIBOND_OCCUPANCY.value
        prop_eng = Property.ANTIBOND_ENERGY_NO_RH.value if no_rh else Property.ANTIBOND_ENERGY.value

        for bond in self.mol.GetBonds():
            found = None
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            atom_symbol_begin = begin_atom.GetSymbol()
            atom_idx_begin = begin_atom.GetAtomMapNum() if no_rh else begin_atom.GetIdx() + 1
            atom_symbol_end = end_atom.GetSymbol()
            atom_idx_end = end_atom.GetAtomMapNum() if no_rh else end_atom.GetIdx() + 1

            pattern_1 = fr"BD\*\(\s+1\)\s*{atom_symbol_begin}\s+{atom_idx_begin}\s+\-\s*{atom_symbol_end}\s+{atom_idx_end}"
            pattern_2 = fr"BD\*\(\s+1\)\s*{atom_symbol_end}\s+{atom_idx_end}\s+\-\s*{atom_symbol_begin}\s+{atom_idx_begin}"

            for line in bond_occupancy_blocks[-1]:
                if re.search(pattern_1, line):
                    bond_occupancy = str.split(line)[-3]
                    bond_energy = str.split(line)[-2]
                    bond.SetProp(prop_occ, bond_occupancy)
                    bond.SetProp(prop_eng, bond_energy)
                    found = True
                    break
                elif re.search(pattern_2, line):
                    bond_occupancy = str.split(line)[-3]
                    bond_energy = str.split(line)[-2]
                    bond.SetProp(prop_occ, bond_occupancy)
                    bond.SetProp(prop_eng, bond_energy)
                    found = True
                    break
            if not found and self.__verbose:
                print(atom_symbol_begin, atom_idx_begin, "-", atom_symbol_end, atom_idx_end, ": Antibond not found")

    def __set_lone_pair_prop(self, no_rh: bool) -> None:
        # FIXME: LP occ and LP eng parameter extraction is not working properly, it returns unwanted chars (")", "d", "%")
        bond_occupancy_blocks = self.__extract_bond_occupancy_blocks(no_rh=no_rh)

        for atom in self.mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_idx = atom.GetAtomMapNum() if no_rh else atom.GetIdx() + 1

            finished = False
            count = 1

            while finished is False:
                pattern = fr"LP\s\(\s+{count}\)\s*{atom_symbol}\s+{atom_idx}"

                # For Rh complexes, no more than 4 LP orbitals will be found
                prop_occ = getattr(
                    Property,
                    f"LONE_PAIR_OCCUPANCY{'_NO_RH' if no_rh else ''}_{min(count, 4)}"
                ).value
                prop_eng = getattr(
                    Property,
                    f"LONE_PAIR_ENERGY{'_NO_RH' if no_rh else ''}_{min(count, 4)}"
                ).value

                match_found = False
                for line in bond_occupancy_blocks[-1]:
                    if re.search(pattern, line):
                        lp_occupancy = str.split(line)[-3]
                        lp_energy = str.split(line)[-2]
                        atom.SetProp(prop_occ, lp_occupancy)
                        atom.SetProp(prop_eng, lp_energy)
                        count += 1
                        match_found = True
                        break
                if not match_found:
                    finished = True

    def __set_lone_pair_orbital_distribution(self) -> None:
        """
        Extract the percentage of orbital occupancy of each lone of the molecule.
        Currently only for no_rh version, meaning that it will only get the values for P atoms.
        """
        orbital_occupancy_blocks = self.__extract_orbital_occupancy_blocks(no_rh=True)

        for atom in self.mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_idx = atom.GetAtomMapNum()

            pattern = f"\s+LP\s+\(\s+1\)\s*{atom_symbol}\s+{atom_idx}\s+s\("
            for line in orbital_occupancy_blocks[-1]:
                if re.search(pattern, line):
                    orbital_occ_raw_list = str.split(line)[-5:]
                    s, p, d = [
                        re.search(r'\d+\.\d+(?=%)', item).group()
                        for item in orbital_occ_raw_list
                        if re.search(r'\d+\.\d+(?=%)', item)
                    ]
                    atom.SetProp(Property.LONE_PAIR_S_ORBITAL_NO_RH.value, s)
                    atom.SetProp(Property.LONE_PAIR_P_ORBITAL_NO_RH.value, p)
                    atom.SetProp(Property.LONE_PAIR_D_ORBITAL_NO_RH.value, d)
                    break

            finished = False
            count = 1

            while finished is False:
                pattern = fr"LP\s\(\s+{count}\)\s*{atom_symbol}\s+{atom_idx}"

                # For Rh complexes, no more than 4 LP orbitals will be found
                prop_occ = getattr(
                    Property,
                    f"LONE_PAIR_OCCUPANCY{'_NO_RH'}_{min(count, 4)}"
                ).value
                prop_eng = getattr(
                    Property,
                    f"LONE_PAIR_ENERGY{'_NO_RH'}_{min(count, 4)}"
                ).value

                match_found = False
                for line in orbital_occupancy_blocks[-1]:
                    if re.search(pattern, line):
                        lp_occupancy = str.split(line)[-3]
                        lp_energy = str.split(line)[-2]
                        atom.SetProp(prop_occ, lp_occupancy)
                        atom.SetProp(prop_eng, lp_energy)
                        count += 1
                        match_found = True
                        break
                if not match_found:
                    finished = True

    def __get_homo_lumo(self, no_rh: bool) -> (float, float):
        orbital_blocks = self.__extract_orbital_blocks(no_rh)
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
        p_list = list(filter(lambda x: x.GetSymbol() == "P", self.mol.GetAtoms()))
        return sorted(p_list, key=lambda x: float(x.GetProp(Property.CHARGE.value)), reverse=True)

    def __get_bridge(self) -> Chem.Atom:
        p1_neighbors = set(map(lambda x: x.GetIdx(), self.p1.GetNeighbors()))
        p2_neighbors = set(map(lambda x: x.GetIdx(), self.p2.GetNeighbors()))
        intersection = list(p1_neighbors.intersection(p2_neighbors))
        intersection_atoms = list(map(lambda x: self.mol.GetAtomWithIdx(x), intersection))
        return next(atom for atom in intersection_atoms if atom.GetSymbol() != "Rh")

    def get_bond_distance(self, atom_1: Chem.Atom, atom_2: Chem.Atom) -> float:
        return rdMolTransforms.GetBondLength(self.__conformer, atom_1.GetIdx(), atom_2.GetIdx())

    def get_angle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom) -> float:
        return rdMolTransforms.GetAngleDeg(self.__conformer, atom_1.GetIdx(), atom_2.GetIdx(), atom_3.GetIdx())

    def get_torsion_angle(self, atom_1: Chem.Atom, atom_2: Chem.Atom, atom_3: Chem.Atom, atom_4: Chem.Atom) -> float:
        return rdMolTransforms.GetDihedralDeg(
            self.__conformer,
            atom_1.GetIdx(),
            atom_2.GetIdx(),
            atom_3.GetIdx(),
            atom_4.GetIdx()
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

    def buried_volume(self, radius: float) -> BuriedVolume:
        atoms = self.mol.GetAtoms()
        elements = [a.GetSymbol() for a in atoms]
        metal_index = self.rh.GetIdx() + 1
        conformer = self.mol.GetConformer()
        coordinates = [np.array([coord.x, coord.y, coord.z]) for coord in
                       (conformer.GetAtomPosition(a.GetIdx()) for a in atoms)]
        excluded_atoms = [at.GetIdx() + 1 for at in self.cod]
        return BuriedVolume(
            elements=elements,
            coordinates=coordinates,
            metal_index=metal_index,
            excluded_atoms=excluded_atoms,
            include_hs=True,
            radius=radius,
            z_axis_atoms=[self.p1.GetIdx() + 1, self.p2.GetIdx() + 1],
            xz_plane_atoms=[self.p2.GetIdx() + 1]
        )

    def get_bond(self, atom_1: Chem.Atom, atom_2: Chem.Atom) -> Chem.Bond:
        return self.mol.GetBondBetweenAtoms(atom_1.GetIdx(), atom_2.GetIdx())

    def __get_p_substituents(self, p: Chem.Atom) -> (Chem.Atom, Chem.Atom):
        """
        Gets the substituents on the P atom, not including the bridge.

        :param p: The phosphorus atom.
        :return: A tuple with the substituent atoms, ordered by the distance to the P-Rh-P plane.
        """
        substituents = list(filter(
            lambda x: x.GetIdx() != self.bridge.GetIdx() and x.GetSymbol() != "Rh" and x.GetSymbol() != "P",
            p.GetNeighbors())
        )
        r1, r2 = sorted(substituents, key=lambda x: self.distance_to_plane(x))
        return r1, r2

    def __get_cod(self) -> list[Chem.Atom]:
        cod = Chem.MolFromSmiles("C1CC=CCCC=C1")
        cyclooctane = Chem.MolFromSmiles("C1CCCCCCC1")  # Pdb conversions shows structure of COD as single bonds

        cod_matches = self.mol.GetSubstructMatches(cod)
        cod_list = []
        if len(cod_matches) == 0:
            cod_matches = self.mol.GetSubstructMatches(cyclooctane)
        # Checks that the COD is coordinated to Rh.
        # Otherwise, it could match a COD or cyclooctane acting as a substituent in the ligand.
        for cod_candidate in cod_matches:
            for carbon_idx in cod_candidate:
                atom = self.mol.GetAtomWithIdx(carbon_idx)
                neighbors = atom.GetNeighbors()
                for neighbor in neighbors:
                    if neighbor.GetSymbol() == "Rh":
                        cod_list.append(cod_candidate)

        cod_atoms = [self.mol.GetAtomWithIdx(c) for c in cod_list[0]]
        for c in cod_atoms:
            cod_hydrogens = [h for h in c.GetNeighbors() if h.GetSymbol() == "H"]
            cod_atoms.extend(cod_hydrogens)

        return cod_atoms

    def __set_no_rh_numbering(self):
        """
        Sets an atom map number mapping each atom with the idx of their no rh version.\n
        This allows to access information of the no rh version without losing the atom reference.
        """
        rh_cod_indices = [self.rh.GetIdx()] + [atom.GetIdx() for atom in self.cod]
        non_rh_cod_indices = [atom.GetIdx() for atom in self.mol.GetAtoms() if atom.GetIdx() not in rh_cod_indices]
        new_order = non_rh_cod_indices + rh_cod_indices
        for i, idx in enumerate(new_order):
            atom = self.mol.GetAtomWithIdx(idx)
            atom.SetAtomMapNum(i + 1)

    def __get_rh_c_substituents(self) -> (Chem.Atom, Chem.Atom, Chem.Atom, Chem.Atom):
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

    def get_prop_wrapper(self, prop_name):
        if not isinstance(prop_name, Property):
            raise TypeError('prop_name must be an instance of Property Enum')

        try:
            return self.GetProp(prop_name.value)
        except KeyError:
            if isinstance(self, Chem.Atom):
                print(
                    f"{bcolors.FAIL}"
                    f"{str.upper(prop_name.value)} not found for atom {self.GetSymbol()} {self.GetIdx()}"
                    f"{bcolors.ENDC}"
                )
            if isinstance(self, Chem.Bond):
                begin_atom_symbol = self.GetBeginAtom().GetSymbol()
                begin_atom_idx = self.GetBeginAtom().GetIdx()
                end_atom_symbol = self.GetEndAtom().GetSymbol()
                end_atom_idx = self.GetEndAtom().GetIdx()
                print(
                    f"{bcolors.FAIL}"
                    f"{str.upper(prop_name.value)}"
                    f" not found for bond {begin_atom_symbol} {begin_atom_idx} - {end_atom_symbol} {end_atom_idx}"
                    f"{bcolors.ENDC}"
                )

    Chem.Atom.get_prop = get_prop_wrapper
    Chem.Bond.get_prop = get_prop_wrapper


class Property(Enum):
    CHARGE = "nbo_charge"
    CHARGE_NO_RH = "nbo_charge_no_rh"
    BOND_OCCUPANCY = "bond_occupancy"
    BOND_OCCUPANCY_NO_RH = "bond_occupancy_no_rh"
    BOND_ENERGY = "bond_energy"
    BOND_ENERGY_NO_RH = "bond_energy_no_rh"
    ANTIBOND_OCCUPANCY = "antibond_occupancy"
    ANTIBOND_OCCUPANCY_NO_RH = "antibond_occupancy_no_rh"
    ANTIBOND_ENERGY = "antibond_energy"
    ANTIBOND_ENERGY_NO_RH = "antibond_energy_no_rh"
    LONE_PAIR_OCCUPANCY_1 = "lone_pair_occupancy_1"
    LONE_PAIR_OCCUPANCY_2 = "lone_pair_occupancy_2"
    LONE_PAIR_OCCUPANCY_3 = "lone_pair_occupancy_3"
    LONE_PAIR_OCCUPANCY_4 = "lone_pair_occupancy_4"
    LONE_PAIR_OCCUPANCY_NO_RH_1 = "lone_pair_occupancy_no_rh_1"
    LONE_PAIR_OCCUPANCY_NO_RH_2 = "lone_pair_occupancy_no_rh_2"
    LONE_PAIR_OCCUPANCY_NO_RH_3 = "lone_pair_occupancy_no_rh_3"
    LONE_PAIR_OCCUPANCY_NO_RH_4 = "lone_pair_occupancy_no_rh_4"
    LONE_PAIR_ENERGY_1 = "lone_pair_energy_1"
    LONE_PAIR_ENERGY_2 = "lone_pair_energy_2"
    LONE_PAIR_ENERGY_3 = "lone_pair_energy_3"
    LONE_PAIR_ENERGY_4 = "lone_pair_energy_4"
    LONE_PAIR_ENERGY_NO_RH_1 = "lone_pair_energy_no_rh_1"
    LONE_PAIR_ENERGY_NO_RH_2 = "lone_pair_energy_no_rh_2"
    LONE_PAIR_ENERGY_NO_RH_3 = "lone_pair_energy_no_rh_3"
    LONE_PAIR_ENERGY_NO_RH_4 = "lone_pair_energy_no_rh_4"
    # LONE_PAIR_S_ORBITAL = "lone_pair_s_orbital"
    # LONE_PAIR_P_ORBITAL = "lone_pair_p_orbital"
    # LONE_PAIR_D_ORBITAL = "lone_pair_d_orbital"
    LONE_PAIR_S_ORBITAL_NO_RH = "lone_pair_s_orbital_no_rh"
    LONE_PAIR_P_ORBITAL_NO_RH = "lone_pair_p_orbital_no_rh"
    LONE_PAIR_D_ORBITAL_NO_RH = "lone_pair_d_orbital_no_rh"
    NMR_ISOTROPIC = "nmr_isotropic"
    NMR_ISOTROPIC_NO_RH = "nmr_isotropic_no_rh"
    NMR_ANISOTROPIC = "nmr_anisotropic"
    NMR_ANISOTROPIC_NO_RH = "nmr_anisotropic_no_rh"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AtomOnPlaneException(Exception):
    pass


if __name__ == "__main__":
    ligand = "l_2"
    comp = ParameterExtractor(path=f"./out/spe/{ligand}_SPE.out", no_rh_path=f"./out/spe_no_rh/{ligand}_SPE_NoRh.out")
