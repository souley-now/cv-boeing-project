This project currently includes public road and lane-detection videos in `data/videos/`:

- `project_video.mp4`
- `challenge_video.mp4`
- `harder_challenge_video.mp4`
- `lane_repo_clip.mp4`
- `aids_dataset_video.mp4`
- `project_video_alt.mp4`

These are suitable inputs for the classical lane-boundary pipeline described in `recommendation.md`.

Why these files were used:

- The document names the TuSimple lane dataset only as an optional benchmark.
- The legacy TuSimple benchmark S3 links published in the original benchmark repo now return a `NoSuchBucket` error.
- The current benchmark page points to Kaggle, but Kaggle CLI credentials are not configured on this machine.

Source repositories:

- `project_video.mp4`: `https://github.com/georgesung/advanced_lane_detection`
- `challenge_video.mp4`, `harder_challenge_video.mp4`, and `project_video_alt.mp4`: `https://github.com/OanaGaskey/Advanced-Lane-Detection`
- `lane_repo_clip.mp4`: `https://github.com/nickorzha/road-lane-detection`
- `aids_dataset_video.mp4`: `https://github.com/apanda-code/Vehicle-Detection-and-Speed-Estimation`
