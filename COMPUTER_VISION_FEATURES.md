# Tracking, segmentacija i Foxglove vizualizacija

Ovaj dokument opisuje dodatne mogućnosti računalnog vida u Jetson anomaly
pipelineu:

- ByteTrack ili BoT-SORT praćenje i stabilni `track_id` tijekom kontinuiranog
  promatranja
- YOLO instance segmentaciju boce
- depth uzorkovanje samo unutar segmentacijske maske
- privacy sliku na kojoj je sve zamućeno osim detektirane boce
- zraku od robota do boce i aproksimacijski krug nesigurnosti u Foxgloveu

Sve mogućnosti mogu se zasebno uključiti ili isključiti kroz YAML konfiguraciju
ili odgovarajuće environment varijable.

## Preporučena potpuna konfiguracija

U `config/anomaly_rosbridge.yaml`:

```yaml
yolo_model_path: yolov8n-seg.pt

tracking_enabled: true
tracking_backend: bytetrack.yaml
tracking_confidence_threshold: 0.25

segmentation_enabled: true
segmentation_depth_mask_erode_px: 2

privacy_image_enabled: true
privacy_image_topic: /anomaly/privacy_image/compressed
privacy_image_publish_hz: 2.0
privacy_blur_kernel_size: 51
privacy_bbox_padding_ratio: 0.03
privacy_use_segmentation_masks: true
privacy_draw_track_id: true
privacy_draw_mask_overlay: true
privacy_mask_overlay_alpha: 0.25

marker_ray_enabled: true
marker_uncertainty_enabled: true
marker_uncertainty_sigma_scale: 2.0
marker_uncertainty_min_radius_m: 0.05
marker_uncertainty_max_radius_m: 1.0
marker_aux_line_width_m: 0.025
```

Iste postavke postoje u
`config/containers/jetson_anomaly.env` kao varijable napisane velikim slovima.
Environment varijable imaju prednost nad YAML vrijednostima.

## Pokretanje

Nakon promjena koje uvode tracker dependency potrebno je ponovno izgraditi
Jetson image:

```bash
docker compose -f docker-compose.yaml -f docker-compose.build.yaml build jetson_anomaly
docker compose up -d --force-recreate jetson_anomaly
```

Direktorij `./models` montira se u container kao `/workspace/models`. Ako
`yolov8n-seg.pt` još ne postoji, Ultralytics ga pri prvom pokretanju preuzima u
taj direktorij. Za prvo pokretanje zato je potrebna mrežna veza. Model nakon
toga ostaje sačuvan na Jetson hostu.

Provjera:

```bash
docker logs -f jetson_anomaly_cont
```

U logu trebaju biti vidljive vrijednosti:

```text
tracking=True
tracker=bytetrack.yaml
segmentation=True
```

Ako segmentacijski model ne vrati maske, pipeline ispisuje upozorenje i nastavlja
s bounding boxovima. Detekcija i event pipeline se zbog toga ne prekidaju.

## ByteTrack i BoT-SORT

Zadani tracker je ByteTrack:

```yaml
tracking_enabled: true
tracking_backend: bytetrack.yaml
```

ByteTrack je preporučeni početni izbor zbog manjeg troška izvođenja. BoT-SORT se
uključuje samo promjenom:

```yaml
tracking_backend: botsort.yaml
```

`tracking_confidence_threshold` je niži od glavnog
`confidence_threshold`. Tracker tako može koristiti slabije detekcije za
održavanje traga, ali event se i dalje stvara samo kada detekcija prijeđe glavni
prag.

`track_id` vrijedi tijekom kontinuiranog traga. Nakon dugog nestanka objekta ili
restarta procesa ID se može promijeniti, zato map-based deduplikacija ostaje
uključena.

Tracking bez segmentacije:

```yaml
yolo_model_path: yolov8n.pt
tracking_enabled: true
segmentation_enabled: false
privacy_use_segmentation_masks: false
```

Potpuno isključivanje trackinga:

```yaml
tracking_enabled: false
```

## Instance segmentacija i depth

Za maske je potreban segmentacijski model, primjerice:

```yaml
yolo_model_path: yolov8n-seg.pt
segmentation_enabled: true
```

Maska ima dvije funkcije:

1. depth vrijednosti uzimaju se samo iz piksela koji pripadaju boci
2. privacy slika otkriva samo siluetu boce umjesto cijelog pravokutnog ROI-ja

`segmentation_depth_mask_erode_px` sužava masku prije čitanja dubine i smanjuje
utjecaj rubnih piksela i pozadine. Vrijednost `0` isključuje eroziju.

## Privacy topic

Topic:

```text
/anomaly/privacy_image/compressed
```

Tip:

```text
sensor_msgs/CompressedImage
```

Kada postoji segmentacijska maska, samo maska boce ostaje vidljiva. Ostatak
slike je zamućen. Ako nema detekcije, zamućena je cijela slika. Na masku se
opcionalno iscrtavaju zelena kontura i `track_id`.

Za prikaz u Foxgloveu:

1. dodati panel **Image**
2. odabrati `/anomaly/privacy_image/compressed`
3. postaviti način skaliranja slike prema potrebi

Bounding-box fallback ostaje aktivan kada maska nije dostupna.

## Zraka i krug nesigurnosti

Oba prikaza objavljuju se unutar postojećeg topica:

```text
/anomaly/markers
```

Tip:

```text
visualization_msgs/MarkerArray
```

MarkerArray sadrži:

- crveni marker objekta
- tekst `bottle #<track_id>`
- narančastu liniju od položaja robota do procijenjenog položaja boce
- žuti krug aproksimacijske nesigurnosti oko boce

Polumjer kruga računa se kao:

```text
distance_uncertainty × marker_uncertainty_sigma_scale
```

i ograničava između konfiguriranog minimalnog i maksimalnog polumjera. Krug
nije potpuna statistička kovarijanca položaja jer robot pose i kut kamere ne
objavljuju uvijek sve potrebne kovarijance; treba ga tumačiti kao vizualnu
procjenu kvalitete lokalizacije.

Za prikaz u Foxgloveu:

1. dodati panel **3D**
2. postaviti fixed frame na `map`
3. uključiti `/anomaly/markers`
4. uključiti `/map` i `/robot_pose_map` radi konteksta

Zraka i krug mogu se neovisno isključiti:

```yaml
marker_ray_enabled: false
marker_uncertainty_enabled: false
```

## Event JSON

Event sadrži dodatna polja:

```json
{
  "track_id": 7,
  "segmentation_mask_used": true,
  "localization": {
    "distance_source": "depth",
    "distance_uncertainty_m": 0.03,
    "bearing_source": "camera_intrinsics"
  }
}
```

Maska se ne sprema u JSON jer bi binarni pikseli nepotrebno povećali događaj.
Spremljena anotirana slika sadrži zelenu masku, bounding box, confidence i
`track_id`.

## Performanse

Ako je inference prespor:

1. ostaviti `tracking_backend: bytetrack.yaml`
2. koristiti `yolov8n-seg.pt`
3. ostaviti `yolo_image_size: 640`
4. smanjiti `privacy_image_publish_hz`
5. postaviti `privacy_draw_mask_overlay: false` ako je JPEG/overlay trošak
   značajan

`inference_every_n_frames` veći od `1` smanjuje opterećenje, ali preskakanje
frameova može pogoršati stabilnost tracker ID-a.

## Važno za automatizirani deploy

`config/anomaly_rosbridge.yaml` i
`config/containers/jetson_anomaly.env` zaštićeni su kroz
`scripts/protected-files.txt`. Automatizirani deploy zato ne prepisuje postojeću
lokalnu konfiguraciju na Jetsonu. Nove vrijednosti treba ručno spojiti u
postojeće datoteke na Jetson hostu.

