# \# Snow-Model: Degree-day factor method



Python codes to calculate and calibrate the degree-day factor method after:



* Martinec J (1960) The degree-day factor for snowmelt runoff forecasting.

IUGG Gen Assem Helsinki, IAHS Publ. 51, IAHS, Wallingford,

UK, pp 468–477



* Çallı SS, Çallı KÖ, Tuğrul Yılmaz M, Çelik M (2022) Contribution

of the satellite-data driven snow routine to a karst hydrological

model. J Hydrol 607:127511. https://doi.org/10.1016/j.jhydrol.2022.127511



\## Calibration\_DDF



This Python code calculates the best-fitting parameters for Tm (threshold temperature \[°C]) and the DDF (degree-day factor \[mm/°C/day]) based on your precipitation, temperature, and snow water equivalent (SWE) data.

The best parameter set is chosen by calculating the Nash–Sutcliffe Efficiency (NSE) between the modelled and the observed SWE. The parameter set with the best NSE is then selected.



\### The Input Parameters are:



* precip\_df : pd.DataFrame

&nbsp;	DataFrame containing precipitation data

* precip\_col : str

&nbsp;	Column name in precip\_df for precipitation values (mm)

* temp\_df : pd.DataFrame

&nbsp;	DataFrame containing temperature data

* temp\_col : str

&nbsp;	Column name in temp\_df for mean temperature values (°C)

* swe\_df : pd.DataFrame

&nbsp;	DataFrame containing observed snow water equivalent data

* swe\_col : str

&nbsp;	Column name in swe\_df for SWE values (mm)

* tm\_lower : float

&nbsp;	Lower limit for threshold temperature Tm (°C)

* tm\_upper : float

&nbsp;	Upper limit for threshold temperature Tm (°C)

* tm\_interval : float

&nbsp;	Interval for testing Tm values

* ddf\_lower : float

&nbsp;	Lower limit for degree-day factor (mm/°C/day)

* ddf\_upper : float

&nbsp;	Upper limit for degree-day factor (mm/°C/day)

* ddf\_interval : float

&nbsp;	Interval for testing DDF values

* plot : bool, default=True

&nbsp;	Whether to generate calibration and comparison plots

* date\_df : pd.DataFrame, optional

&nbsp;	DataFrame containing date information for time-series plots

* date\_col : str, optional

&nbsp;	Column name in date\_df for dates (will be converted to datetime if not already)



\### The Output is:



* best\_tm : float  

&nbsp;   Optimal threshold temperature (°C)  

* best\_ddf : float  

&nbsp;   Optimal degree-day factor (mm/°C/day)  

* best\_nse : float  

&nbsp;   Best Nash–Sutcliffe Efficiency achieved  

* results\_df : pd.DataFrame  

&nbsp;   DataFrame with NSE values for all parameter combinations





You need daily precipitation, temperature, and SWE data (all with exactly the same length and no data gaps) to run this code. You can choose the upper and lower limits and the interval of the Tm and DDF values that should be tested.

You get the best values for Tm and DDF. You also get the best achieved NSE and a DataFrame with all parameter sets tested and the resulting NSE values. You get a plot showing the calibration process, with the best parameters marked (if plot=True). Additionally, you get a plot comparing the modelled SWE and the observed SWE for the best parameter sets (if plot=True).





