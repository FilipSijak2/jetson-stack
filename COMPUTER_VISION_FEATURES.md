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
marker_ray_ttl_s: 2.0
marker_uncertainty_enabled: true
marker_uncertainty_sigma_scale: 2.0
marker_uncertainty_min_radius_m: 0.05
marker_uncertainty_max_radius_m: 1.0
marker_aux_line_width_m: 0.025

detection_3d_enabled: true
detection_3d_topic: /anomaly/detections_3d
detection_3d_frame_id: ""
detection_3d_require_mask: true
detection_3d_publish_hz: 5.0
detection_3d_ttl_s: 0.75
detection_3d_min_valid_points: 30
detection_3d_lower_percentile: 5.0
detection_3d_upper_percentile: 95.0
detection_3d_sample_stride: 2
detection_3d_minimum_thickness_m: 0.05
detection_3d_line_width_m: 0.01
detection_3d_text_enabled: true
detection_3d_text_height_m: 0.035
detection_3d_text_show_label: false
detection_3d_text_show_confidence: false
detection_3d_text_show_distance: true
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

Aktivni RGB, CameraInfo i depth ulazi dolaze iz iste RealSense kamere:

```yaml
camera_topic: /camera/realsense/color/image_raw/compressed
camera_info_topic: /camera/realsense/color/camera_info
depth_topic: /camera/realsense/aligned_depth_to_color/image_raw/compressedDepth
```

Za depth se koristi `compressedDepth` transport kako se sirova 16-bitna slika
ne bi prenosila kao veliki JSON/base64 payload kroz rosbridge. Dekodirana
vrijednost `16UC1` pretvara se iz milimetara u metre. Isti RealSense izvor važan
je i zbog vremenske sinkronizacije i zato što se segmentacijska maska iz RGB
slike primjenjuje na aligned-depth piksele.

Depth se kroz rosbridge ograničava na najviše 10 Hz
(`depth_throttle_ms: 100`), a RGB-depth vremenska razlika mora biti najviše
`0.35 s`. Ta tolerancija pokriva transportno i inferencijsko kašnjenje izmjereno
na platformi, dok `depth_max_age_s: 1.0` i dalje odbacuje stvarno zastarjele
frameove.

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

## Slike za dokumentaciju

Postavka:

```yaml
save_documentation_images: true
```

odnosno environment varijabla:

```text
SAVE_DOCUMENTATION_IMAGES=1
```

za svaki potvrđeni događaj sprema sinkronizirani skup u
`/home/jetson/anomaly_logs/images/documentation/`:

1. privacy RGB kadar s bounding boxom: pozadina je zamućena, a segmentirani
   objekt ostaje vidljiv bez obojenog mask overlayja
2. izvornu binarnu segmentacijsku masku
3. masku erodiranu prema `segmentation_depth_mask_erode_px`
4. valjane aligned-depth piksele unutar erodirane maske i središnjeg ROI-ja,
   obojene TURBO paletom (bliže crveno, dalje plavo)
5. završni kompozit prethodna četiri prikaza u rasporedu 2 × 2, s oznakama
   `(a)`–`(d)` i nazivom svakog prikaza

Datoteke istog događaja dijele prefiks, primjerice
`anom_00042_bottle_01_rgb_bbox.jpg` do
`anom_00042_bottle_04_depth_colormap.png`. Kada je skup potpun, automatski se
sprema i `anom_00042_bottle_05_documentation_composite.png`, spreman za
umetanje u Word ili PDF. Ako maska ili vremenski usklađen depth kadar nisu
dostupni, spremaju se dostupni pojedinačni prikazi i u log se upisuje koji
prikazi nedostaju; nepotpuni kompozit se ne izrađuje.

Dnevna deduplikacija eventa ne blokira dokumentacijsko snimanje. Kada je objekt
već prijavljen (`already_reported_today`) ili je aktivan cooldown, prvi frame
tog tracka koji ima segmentacijsku masku i valjani sinkronizirani depth sprema
se jednom s prefiksom poput
`capture_bottle_track_43_20260719_133012_417`. Tako se četiri pojedinačna
prikaza i `05_documentation_composite.png` mogu dobiti bez premještanja boce i
bez stvaranja duplikata u `events.jsonl`.

Privacy RGB prikaz koristi iste postavke kao privacy stream:
`privacy_blur_kernel_size`, `privacy_bbox_padding_ratio` i
`privacy_use_segmentation_masks`. U aktivnoj konfiguraciji segmentacijska maska
određuje jedino područje koje ostaje nezamućeno.

### Slika segmentacije i trackinga kroz tri framea

Postavka:

```yaml
save_tracking_documentation_sequence: true
```

odnosno:

```text
SAVE_TRACKING_DOCUMENTATION_SEQUENCE=1
```

automatski sprema jedan vodoravni niz od tri uzastopna inferencijska framea za
svaki `track_id`. Svaki panel prikazuje privacy RGB pozadinu, zelenu
segmentacijsku masku i konturu, bounding box, confidence, procijenjenu
udaljenost i njezin izvor (`depth`, `laser` ili `default`). Paneli su označeni
kao `(a) Frame t-2`, `(b) Frame t-1` i `(c) Frame t`.

Datoteka se sprema u
`/home/jetson/anomaly_logs/images/documentation/` pod nazivom poput:

```text
tracking_bottle_track_7_20260719_143025_418_three_frame_sequence.png
```

Sekvenca se izrađuje samo kada sva tri framea imaju isti track ID i
segmentacijsku masku. Ako objekt nedostaje u jednom obrađenom frameu, povijest
tog tracka počinje ispočetka. Za svaki track tijekom jednog pokretanja procesa
sprema se samo prvi potpuni primjer kako se ne bi nepotrebno punio disk.

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

Zraka je prikaz trenutnog opažanja, a ne trajna veza robota i anomalije.
Obnavlja se dok se boca detektira i nestaje nakon `marker_ray_ttl_s` sekundi
kada opažanje prestane. Objektni marker i krug ostaju do `marker_ttl_s`.

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

## Live 3D bounding box boce

Live 3D kutija objavljuje se na:

```text
/anomaly/detections_3d
```

Tip poruke je `visualization_msgs/MarkerArray`. Pipeline:

1. uzima segmentacijsku masku boce
2. bira vremenski najbliži aligned-depth frame
3. deprojicira depth piksele pomoću `fx`, `fy`, `cx` i `cy`
4. odbacuje rubne i pozadinske vrijednosti robusnim percentilima
5. objavljuje zeleni wireframe s 12 bridova i tekstom
   `bottle #<track_id> <confidence> <distance>m`

Marker se objavljuje u frameu iz RGB headera, normalno
`realsense_color_optical_frame`. `detection_3d_frame_id` se koristi samo kao
ručni override. Za prikaz u Foxgloveu:

1. otvoriti panel **3D**
2. uključiti `/anomaly/detections_3d`
3. ostaviti fixed frame `map` ako postoji TF do color optical framea
4. ako Foxglove prijavi nepoznat frame, provjeriti `/tf_static` ili privremeno
   postaviti fixed frame na vrijednost iz RGB headera

`detection_3d_require_mask: true` sprječava stvaranje nepouzdanog volumena iz
cijelog bounding boxa. Postavljanje na `false` omogućuje bbox fallback.

Širina i visina predstavljaju robusne granice vidljivih 3D piksela. Kamera ne
vidi stražnju stranu boce, pa je `size_z` samo raspon vidljive dubine uz
minimalnu vizualnu debljinu `detection_3d_minimum_thickness_m`; nije potpuna
rekonstrukcija fizičkog volumena.

Isključivanje:

```yaml
detection_3d_enabled: false
```

## Dnevna deduplikacija i daily mapa

`events.jsonl` ostaje trajni zapis svih događaja, ali se za deduplikaciju pri
pokretanju učitavaju samo eventi iz trenutnog lokalnog dana. Zato se ista boca
može ponovno evidentirati sljedećeg dana, a
`map_images/daily/anomalies_YYYY-MM-DD.png` sadrži samo anomalije tog dana.

Na prijelazu preko ponoći pipeline automatski:

- prazni dnevnu listu već prijavljenih lokacija
- prazni pending opažanja i cooldown
- briše stare Foxglove markere
- započinje novu daily map sliku

Globalni `anom_XXXXX` brojač i stari redci u `events.jsonl` ostaju sačuvani.

Log sada navodi razlog zbog kojeg detekcija još nije proizvela event:

```text
event_gate=pending_1_of_2
event_gate=already_reported_today
event_gate=cooldown
```

Uspješan event ispisuje putanje originalne slike, anotirane slike i daily
karte. Lokacija se označava prijavljenom tek nakon uspješnog zapisa eventa, pa
greška pri spremanju ne blokira sljedeći pokušaj.

## Automatski prilazak i kvalitetna privacy fotografija

Opcionalni inspection workflow povezuje potvrdenu YOLO detekciju s Nav2
navigacijom:

1. detekcija mora proci postojeca `anomaly_min_observations: 2`
1. Jetson prihvaca samo depth/laser lokalizaciju s nesigurnoscu do `0.30 m`
1. Raspberry racuna goal na zadanoj standoff udaljenosti, okrenut prema boci
1. zahtjev se odbija ako je robot u manual modu ili izvrsava drugi goal
1. nakon dolaska robot miruje `1 s`
1. Jetson bira najoštriji od 8 privacy kadrova
1. spremljena lokacija se vise ne pregledava isti dan

Funkcija je zadano iskljucena. Za ukljucivanje na Jetsonu postaviti u
`config/anomaly_rosbridge.yaml`:

```yaml
inspection_enabled: true
inspection_standoff_m: 0.70
inspection_min_distance_m: 0.40
inspection_max_distance_m: 3.0
inspection_max_uncertainty_m: 0.30
inspection_require_metric_distance: true
inspection_capture_frames: 8
inspection_capture_timeout_s: 8.0
inspection_request_timeout_s: 70.0
inspection_jpeg_quality: 95
inspection_once_per_cluster: true
inspection_retry_cooldown_s: 60.0
inspection_group_enabled: true
inspection_group_radius_m: 2.0
inspection_group_collection_s: 0.75
inspection_group_min_objects: 2
inspection_group_max_objects: 10
inspection_group_fov_margin_ratio: 1.25
inspection_group_max_standoff_m: 2.50
inspection_group_require_all_visible: true
```

Budući da environment ima prednost, u
`config/containers/jetson_anomaly.env` treba postaviti:

```env
INSPECTION_ENABLED=1
```

Na Raspberryju u `config/containers/nav_cont.env`:

```env
ENABLE_ANOMALY_INSPECTION=1
INSPECTION_ONLY_WHEN_IDLE=true
INSPECTION_DEFAULT_STANDOFF_M=0.70
```

Inspection koristi topice:

- `/anomaly/inspection/request`
- `/anomaly/inspection/status`
- `/anomaly/inspection/result`
- `/anomaly/inspection/privacy_image/compressed`

Snimke se spremaju u
`anomaly_logs/images/inspection/<datum>/`, a rezultati u
`anomaly_logs/inspections.jsonl`. Prebacivanje na joystick tijekom prilaska
odmah otkazuje inspection goal.

Jetson nakon potvrde kratko prikuplja kandidate tijekom `0.75 s`. Ako su
najmanje dvije boce medusobno udaljene najvise `2.0 m`, salje ih kao jedan
group inspection. Goal gleda prema sredistu grupe, a
standoff se automatski povecava prema rasponu grupe i horizontalnom FOV-u kamere
(najvise do `2.50 m`). Snima se jedna privacy fotografija i svih 2-10 ciljeva
mora biti vidljivo u odabranom kadru. Nakon uspjeha sve boce iz grupe oznacavaju
se pregledanima, pa robot ne radi poseban prilazak svakoj.

Ako Nav2 ne moze planirati ili izvrsiti siguran prilazak, Raspberry uz
`INSPECTION_CAPTURE_ON_NAV_FAILURE=true` ne zaobilazi costmap niti smanjuje
sigurnosne margine. Zaustavlja pokusaj navigacije, ceka stabilizaciju i trazi
grupnu fotografiju iz trenutne pozicije. Ako sve boce nisu vidljive, capture
zavrsava neuspjehom i moze se ponoviti nakon cooldowna.

Za stvarno kvalitetniji ulaz Raspberry konfiguracija koristi
`RS_COLOR_PROFILE=640x480x15` i `RS_COMPRESSED_JPEG_QUALITY=75`; konačna
inspection fotografija sprema se s JPEG kvalitetom 95.

## Odvajanje vise boca i citljiviji markeri

Razliciti aktivni `track_id`-evi vise se ne spajaju u srednju lokaciju. Zadani
prostorni pragovi sada su:

```yaml
cluster_merge_radius_m: 0.25
marker_association_radius_m: 0.40
reported_merge_radius_m: 0.45
tracked_object_min_separation_m: 0.01
```

3D box svake boce dobiva stabilnu boju prema `track_id`-u. Tekst je smanjen s
punog `bottle #149 0.83 0.75m` na citljiviji `#149 · 0.75 m`. Map marker koristi
samo `#track_id`, pa se susjedne boce lakse razlikuju u Foxgloveu.

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

## Automatska evaluacija bez anotiranja dataseta

Pipeline jednom u sekundi sprema lagani performance uzorak u:

```text
anomaly_logs/evaluation/performance.jsonl
```

Biljeze se decode vrijeme, YOLO inference vrijeme, efektivni camera FPS, broj
detekcija, tracking coverage i segmentation coverage. Postojeci
`events.jsonl` ostaje nepromijenjen.

Jedini rucni korak je oznacavanje rijetkog false positive eventa. Za pregled
jucerasnjih detekcija:

```bash
python3 scripts/mark_event.py --list --date yesterday
```

Oznacavanje poznatog ID-a:

```bash
python3 scripts/mark_event.py anom_00042 --verdict FP
```

Ako je FP bio zadnji jucerasnji event:

```bash
python3 scripts/mark_last_event_fp.py --date yesterday
```

Review se append-only sprema u `anomaly_logs/event_reviews.jsonl`. Originalni
event i slike se ne mijenjaju. Pogresna oznaka ispravlja se novom naredbom s
`--verdict TP`; izvjestaj uvijek koristi zadnju oznaku za event.

Izvjestaj se generira:

```bash
python3 scripts/generate_cv_report.py
```

Rezultati su:

```text
anomaly_logs/evaluation/latest/report.html
anomaly_logs/evaluation/latest/summary.json
anomaly_logs/evaluation/latest/events.csv
```

Alati rade i nad starim eventima dok god su njihovi redci jos u
`events.jsonl`. Za povijesne evente bit ce dostupne event/localization metrike;
FPS i inference latency postoje tek od verzije pipelinea koja zapisuje
`performance.jsonl`.

Bez ground truth anotacija precision je procjena u kojoj se neoznaceni eventi
tretiraju kao stvarne detekcije. Pravi recall, F1, confusion matrix,
segmentation IoU i apsolutna localization pogreska namjerno se ne prikazuju kao
ground-truth metrike. Report umjesto toga daje jasno oznacene threshold
proxy-vrijednosti.

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
