# SunergyCityJSON
SunergyCityJSON is a Python-based framework enabled by CityJSON designed to jointly analyze publicly available 3D building models, incorporate windows, and perform solar radiation and shadow analysis, as well as heating and cooling demand calculations.

# Features
- Shadow analysis, and solar radiation calculation with the possibility to export shapfile for 3D visualization.
<p align="center">
  <img src="https://github.com/user-attachments/assets/cf8ba444-dd33-4de7-8e3e-b2845cd69e8e" alt="Shadow height" width="40%">
  <img src="https://github.com/user-attachments/assets/07692f76-f24d-4099-b02c-02bb0908a0ae" alt="Solar radiation" width="40%">
</p>


- Obtaining information on solar heat gain by individual walls.
<p align="center">
  <img src="https://github.com/user-attachments/assets/3ce7eae3-4c89-443b-bbfe-3d300713a265" alt="Image 1" width="45%">
  <img src="https://github.com/user-attachments/assets/932aebf7-4eac-4747-8be0-0fdcd9262f1c" alt="Image 2" width="45%">
</p>

# Run 
1. With your own data  
   To run the tool you need to prepare you input data, identify ID of the desired building (An attribute in Citygml / CityJSON file) and optionally indicate the an hour where you would like to export shapefile for 3D visulaization with tools such as ArcGISPro.  
`python3 SunergyCityJSON.py "cityjsonfilename.json" "building_id" hour`  
   
3. Test with provided input  
   `python3 SunergyCityJSON.py "684_5334.json" "DEBY_LOD2_4615927" 6512`




# Instruction on input data 
- CityJSON file of the 3D building model: 
  - Download open-access LoD2 data of Bavaria in Citygml format from https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2
  - Convert citygml to cityJSON using https://github.com/citygml4j/citygml-tools. After clonning the reoprository you may use the command line: `citygml-tools to-cityjson --cityjson-version=1.0 /path/to/your/CityGML/file.gml`

- Year of construction  
  Building age is used as an indicator for the thrermal properties of the building enevelope which is then used for heat transfer calculation and the estimation of heating and cooling demand. Therefor, a shapefile including buildings' foortprints and the corresponsing year of construction is required as an input.  
    
- Yearly outside temperature data


  





