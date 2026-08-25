import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "notebooks" / "AUX-BB - Baseline-Centered Fixed Unified ESN Analysis.ipynb"
notebook = json.loads(path.read_text())

old = '''    ax_heatmap.set(title="Full reservoir-state heatmap", ylabel="Reservoir unit")
    fig.colorbar(image, ax=ax_heatmap, label="Activation", pad=0.01)
'''
new = '''    ax_heatmap.set(title="Full reservoir-state heatmap", ylabel="Reservoir unit")

    # An inset colorbar does not shrink the heatmap's plotting axis.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colorbar_axis = inset_axes(
        ax_heatmap,
        width="1.4%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0, 1.0, 1.0),
        bbox_transform=ax_heatmap.transAxes,
        borderpad=0,
    )
    fig.colorbar(image, cax=colorbar_axis, label="Activation")
'''

matches = 0
for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if old in source:
        cell["source"] = source.replace(old, new).splitlines(keepends=True)
        matches += 1

if matches != 1:
    raise RuntimeError(f"Expected one heatmap colorbar block, found {matches}.")

path.write_text(json.dumps(notebook, indent=1))
print(path)
