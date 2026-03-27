from helper_plotting import plot_combined_pareto_with_tradeoff

JOINT = "PIP"
name = "00_dxf_04_dxf2_03"

plot_combined_pareto_with_tradeoff(
     f"sim_results/{JOINT}/{name}/standard_single_mag.csv",
     None, #f"sim_results/{JOINT}/08_dxf_04_dxf2_03/standard.csv",
     weight_obj1=1.0,
     weight_obj2=0.5,
     show_opt=True
 )