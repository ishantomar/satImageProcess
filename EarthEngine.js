var startDate = "2022-05-01"
var endDate = "2023-05-01"
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(startDate, endDate).filterBounds(roi)
var dwImage = ee.Image(dw.mosaic()).clip(roi)
print('Test', dwImage)
var CLASS_NAMES = [
    'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];

var VIS_PALETTE = [
    '419bdf', '397d49', '88b053', '7a87c6', 'e49635', 'dfc35a', 'c4281b',
    'a59b8f', 'b39fe1'];
var classification = dwImage.select('label')
Map.addLayer(classification, {min:0, max:8, palette:VIS_PALETTE}, "Classified Image")
Map.centerObject(roi)
Export.image.toDrive({
  image: classification,
  description: "Dynamic_WorldLULC1_2022-05-01_2022-11-01",
  scale: 10,
  region: roi,
  maxPixels: 1e13
});
