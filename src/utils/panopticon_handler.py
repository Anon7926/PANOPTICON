from datasets import load_from_disk

ds = load_from_disk("PANOPTICON")
print(ds[::4837]['Prompt'])
