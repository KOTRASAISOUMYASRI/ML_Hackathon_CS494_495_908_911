"""Development-only Folium export including reference geometry after inference."""
import folium
from data_loader import load_trip
from route_recovery.main import MatchRequest, match, render_map, inventory


def main():
    request=MatchRequest(trip_id='00000052/00000052',evaluate=True,diagnose=True)
    result=match(request); view=render_map(result,request)
    # Controls must be rendered after every feature group has been defined.
    for key,value in list(view._children.items()):
        if isinstance(value,folium.LayerControl): del view._children[key]
    reference=load_trip(inventory('student_test')[request.trip_id],with_reference=True)
    layer=folium.FeatureGroup(name='Reference road arcs red dashed')
    for arc in reference.reference_arcs:
        u,v=reference.arcs[arc]
        folium.PolyLine([reference.nodes[u][::-1].tolist(),reference.nodes[v][::-1].tolist()],color='red',weight=3,dash_array='6,5',opacity=.8).add_to(layer)
    layer.add_to(view)
    folium.LayerControl(collapsed=False).add_to(view)
    view.save('reports/submission/folium_trip52.html')


if __name__=='__main__': main()
