from fafbseg import flywire
import navis

flywire.set_default_dataset("public")

n = flywire.get_mesh_neuron(720575940621039145)
print(n)
navis.plot3d(n, backend="plotly")

fig = navis.plot3d(n, backend="plotly", inline=False)
fig.show()

