import numpy as np
from rdkit import Chem
import pandas as pd
# 读取 .npz 文件
data = np.load('qm9gen.npz')
molecule_data = data['samples']
# 提取前两个通道的数据
channel_1 = molecule_data[:, 0, :, :]  # 第一个通道（XYZ坐标）
channel_2 = molecule_data[:, 1, :, :]  # 第二个通道（原子类型）

# 四舍五入操作
channel_2 = np.round(channel_2)

# 删除每个分子中全0的行，并且删除对应的第一通道（XYZ坐标）行
channel_2_non_zero = [molecule[molecule.any(axis=1)] for molecule in channel_2]
channel_1_non_zero = [channel_1[i][molecule.any(axis=1)] for i, molecule in enumerate(channel_2)]

# 原子类型映射
def get_atom_type(atom_str):
    atom_map = {
        100: 'C', 110: 'H', 10: 'N', 1: 'O', 111: 'F'
    }
    return atom_map.get(int(atom_str), 'Unknown')

# 计算原子之间的距离
def compute_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

# 键长字典
# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}


def print_table(bonds_dict):
    letters = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Si', 'Cl', 'Br', 'I']

    new_letters = []
    for key in (letters + list(bonds_dict.keys())):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f'{x} & ', end='')
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f'{y} & ', end='')
            if x in bonds_dict[y]:
                print(f'{bonds_dict[y][x]} & ', end='')
            else:
                print('- & ', end='')
        print()


# print_table(bonds3)


def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError('Not in dict ' + str((atom1, atom2)))

                assert bond == bond_check, (
                    f'{bond} != {bond_check} for {atom1}, {atom2}')


stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}

def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # 调整单位

    if distance < bonds1[atom1][atom2]:
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            if distance < bonds2[atom1][atom2]:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    if distance < bonds3[atom1][atom2]:
                        return 3  # 三键
                return 2  # 双键
        return 1  # 单键
    return 0  # 无键

def print_bond_counts_and_stability(molecule_data, xyz_data, allowed_bonds):
    stable_atoms = 0
    total_atoms = 0
    unstable_atoms = 0  # 用于记录不稳定原子的数量

    for molecule_idx, molecule in enumerate(molecule_data):
        xyz = xyz_data[molecule_idx]
        num_atoms = molecule.shape[0]

        # 检查该分子中是否包含无法识别的原子类型
        if any(get_atom_type(''.join(molecule[atom_idx].astype(int).astype(str))) == 'Unknown' for atom_idx in range(num_atoms)):
            #print(f"分子 {molecule_idx} 包含无法识别的原子类型，跳过此分子。")
            continue  # 如果有无法识别的原子，跳过此分子

        bond_counts = {i: 0 for i in range(num_atoms)}

        for atom1_idx in range(num_atoms):
            for atom2_idx in range(atom1_idx + 1, num_atoms):
                atom1 = xyz[atom1_idx]
                atom2 = xyz[atom2_idx]
                distance = compute_distance(atom1, atom2)

                atom1_type = get_atom_type(''.join(molecule[atom1_idx].astype(int).astype(str)))
                atom2_type = get_atom_type(''.join(molecule[atom2_idx].astype(int).astype(str)))

                bond_order = get_bond_order(atom1_type, atom2_type, distance)

                if bond_order > 0:
                    bond_counts[atom1_idx] += 1
                    bond_counts[atom2_idx] += 1

        for atom_idx in range(num_atoms):
            atom_type = get_atom_type(''.join(molecule[atom_idx].astype(int).astype(str)))
            total_atoms += 1
            total_bonds = bond_counts[atom_idx]

            # 计算原子的最大允许键数
            allowed_bond_count = allowed_bonds.get(atom_type)
            if isinstance(allowed_bond_count, list):
                allowed_bond_count = max(allowed_bond_count)  # 使用最大值

            if total_bonds <= allowed_bond_count:
                stable_atoms += 1
                #print(f"分子 {molecule_idx}, 原子 {atom_idx} ({atom_type}) - 总键连接数: {total_bonds}, 最大允许键数: {allowed_bond_count}, 稳定性: 稳定")
            else:
                unstable_atoms += 1
                #print(f"分子 {molecule_idx}, 原子 {atom_idx} ({atom_type}) - 总键连接数: {total_bonds}, 最大允许键数: {allowed_bond_count}, 稳定性: 不稳定")

    stability_ratio = stable_atoms / total_atoms if total_atoms > 0 else 0
    print(f"Atom stable(%): {stability_ratio:.4f} ")

# 调用打印函数来显示每个原子的稳定性
print_bond_counts_and_stability(channel_2_non_zero, channel_1_non_zero, allowed_bonds)

def check_molecule_stability(molecule, xyz_data, allowed_bonds):
    num_atoms = molecule.shape[0]
    bond_counts = {i: 0 for i in range(num_atoms)}

    # 检查是否存在无法识别的原子类型
    for atom_idx in range(num_atoms):
        atom_type = get_atom_type(''.join(molecule[atom_idx].astype(int).astype(str)))
        if atom_type == 'Unknown':
            return False  # 遇到无法识别的原子类型，直接认为不稳定

    for atom1_idx in range(num_atoms):
        for atom2_idx in range(atom1_idx + 1, num_atoms):
            atom1 = xyz_data[atom1_idx]
            atom2 = xyz_data[atom2_idx]
            distance = compute_distance(atom1, atom2)

            atom1_type = get_atom_type(''.join(molecule[atom1_idx].astype(int).astype(str)))
            atom2_type = get_atom_type(''.join(molecule[atom2_idx].astype(int).astype(str)))

            bond_order = get_bond_order(atom1_type, atom2_type, distance)

            if bond_order > 0:
                bond_counts[atom1_idx] += 1
                bond_counts[atom2_idx] += 1

    # 检查每个原子的稳定性
    for atom_idx in range(num_atoms):
        atom_type = get_atom_type(''.join(molecule[atom_idx].astype(int).astype(str)))
        total_bonds = bond_counts[atom_idx]

        # 获取该原子的最大允许键数
        allowed_bond_count = allowed_bonds.get(atom_type)
        if isinstance(allowed_bond_count, list):
            allowed_bond_count = max(allowed_bond_count)  # 使用最大值

        if total_bonds > allowed_bond_count:
            return False  # 只要一个原子不稳定，整个分子不稳定

    return True  # 如果所有原子稳定，分子稳定

def compute_molecular_stability(channel_2_non_zero, channel_1_non_zero, allowed_bonds):
    stable_molecules = 0
    total_molecules = len(channel_2_non_zero)

    for molecule, xyz_data in zip(channel_2_non_zero, channel_1_non_zero):
        if check_molecule_stability(molecule, xyz_data, allowed_bonds):
            stable_molecules += 1

    molecular_stability = stable_molecules / total_molecules if total_molecules > 0 else 0
    print(f"Mol stable(%)：{molecular_stability:.4f} ")


# 计算并输出分子稳定性
compute_molecular_stability(channel_2_non_zero, channel_1_non_zero, allowed_bonds)
# 获取每个原子的稳定性，判断是否满足 allowed_bonds 限制

def get_smiles_from_molecule(molecule, xyz_data):
    # 假设你已经有了原子类型和坐标数据，并且能成功构建一个分子
    atom_types = [get_atom_type(''.join(molecule[atom_idx].astype(int).astype(str))) for atom_idx in range(molecule.shape[0])]

    # 检查是否存在无法识别的原子类型
    if any(atom_type == 'Unknown' for atom_type in atom_types):
        return None  # 如果有无法识别的原子，跳过该分子的SMILES生成

    # 创建 RDKit 分子对象
    mol = Chem.RWMol()

    atom_map = {}  # 用来映射原子到分子中的位置

    # 添加原子
    for idx, atom_type in enumerate(atom_types):
        rd_atom = Chem.Atom(atom_type)
        atom_idx = mol.AddAtom(rd_atom)
        atom_map[idx] = atom_idx

    # 添加键（使用 get_bond_order 函数来判断键类型）
    num_atoms = len(atom_types)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom1 = xyz_data[i]
            atom2 = xyz_data[j]
            distance = compute_distance(atom1, atom2)

            atom1_type = atom_types[i]
            atom2_type = atom_types[j]

            bond_order = get_bond_order(atom1_type, atom2_type, distance)

            if bond_order > 0:
                # 使用 get_bond_order 返回的键类型来添加键
                if bond_order == 1:
                    mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)
                elif bond_order == 2:
                    mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.DOUBLE)
                elif bond_order == 3:
                    mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.TRIPLE)

    # 生成 SMILES
    smiles = Chem.MolToSmiles(mol)

    # 检查 SMILES 是否有效并且不在训练集中
    if smiles:
        return smiles
    else:
        return None

def compute_valid_and_unique_ratio(channel_2_non_zero, channel_1_non_zero):
    all_smiles = []  # 用来存储所有生成的 SMILES
    valid_smiles = set()  # 用来存储有效的 SMILES集合
    unique_smiles = set()  # 用来存储唯一的 SMILES集合
    total_molecules = len(channel_2_non_zero)  # 总分子数量

    # 遍历每个分子
    for molecule, xyz_data in zip(channel_2_non_zero, channel_1_non_zero):
        # 生成 SMILES
        smiles = get_smiles_from_molecule(molecule, xyz_data)

        if smiles:  # 如果生成了有效的 SMILES
            all_smiles.append(smiles)
            valid_smiles.add(smiles)  # 将有效的 SMILES 添加到集合中

    # 统计唯一的 SMILES
    smile_counts = pd.Series(all_smiles).value_counts()  # 统计每个 SMILES 出现的次数
    unique_smiles = {smiles for smiles, count in smile_counts.items() if count == 1}  # 只保留出现一次的 SMILES

    # 计算有效分子占比
    valid_ratio = len(valid_smiles) / total_molecules if total_molecules > 0 else 0
    # 计算唯一 SMILES 占比
    unique_ratio = len(unique_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0

    # 输出结果
    print(f"Valid(%): {valid_ratio:.4f}")
    print(f"Valid and Unique(%): {unique_ratio:.4f}")


# 调用该函数
compute_valid_and_unique_ratio(channel_2_non_zero, channel_1_non_zero)

def compute_smiles_not_in_dataset_ratio(channel_2_non_zero, channel_1_non_zero, smiles_dataset,
                                        output_file='capgenring.txt'):
    stable_molecules = 0
    not_in_dataset_molecules = 0
    valid_smiles = set()  # 用于存储唯一有效的 SMILES
    total_molecules = len(channel_2_non_zero)

    # 将数据集中的所有 SMILES 存储在集合中，快速查找
    dataset_smiles = set(smiles_dataset.iloc[:, 1].dropna())  # 第二列的 SMILES

    # 打开输出文件准备写入 SMILES
    with open(output_file, 'w') as f:
        f.write("Generated SMILES (not in dataset)\n")  # 可选：写入标题

        for molecule, xyz_data in zip(channel_2_non_zero, channel_1_non_zero):
            # 判断分子是否稳定
            if check_molecule_stability(molecule, xyz_data, allowed_bonds):
                stable_molecules += 1

                # 判断 SMILES 是否有效
                smiles = get_smiles_from_molecule(molecule, xyz_data)

                if smiles:  # 如果生成有效的 SMILES
                    valid_smiles.add(smiles)  # 将有效的 SMILES 添加到集合中

                    # 检查生成的 SMILES 是否存在于数据集中
                    if smiles not in dataset_smiles:
                        not_in_dataset_molecules += 1
                        # 将 SMILES 写入到文件
                        f.write(smiles + '\n')  # 每个 SMILES 单独一行

    # 计算不在数据集中的 SMILES 的比例
    not_in_dataset_ratio = not_in_dataset_molecules / stable_molecules if stable_molecules > 0 else 0
    print(f"Novelty(%)：{not_in_dataset_ratio:.4f} ")



# 调用函数来计算不在数据集中的 SMILES 的比例，并保存到文本文件
# 假设 smiles_dataset 是从 csv 文件中读取的 DataFrame

smiles_dataset = pd.read_csv("qm9_dataset.csv")  # 读取包含 SMILES 数据集的 CSV 文件
compute_smiles_not_in_dataset_ratio(channel_2_non_zero, channel_1_non_zero, smiles_dataset,
                                    output_file='capgenring.txt')
