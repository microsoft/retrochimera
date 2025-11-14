import random


def get_product_roots(product_atom_ids: list[int], num_augmentations: int) -> list[int]:
    product_roots = [-1]

    if len(product_atom_ids) < num_augmentations:
        product_roots.extend(product_atom_ids)
        product_roots.extend(
            random.choices(product_roots, k=num_augmentations - len(product_roots))
        )
    else:
        product_roots.extend(random.sample(product_atom_ids, num_augmentations - 1))

    assert len(product_roots) == num_augmentations
    return product_roots
