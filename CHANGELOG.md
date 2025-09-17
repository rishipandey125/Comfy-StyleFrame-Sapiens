# Changelog

This is an automated changelog based on the commits in this repository.
## [main] - 2025-08-08

### Bug Fixes

- 🐛 I/O logic ([b265b3b](https://github.com/melMass/ComfyUI-SapiensPose/commit/b265b3b3ac44f58099b2013ea95e354994eb4eb6))

   1.0.0 was batching the keypoints incorectly

- 🐛 change default chunk size ([c1d22f3](https://github.com/melMass/ComfyUI-SapiensPose/commit/c1d22f369c730b97628731d8bfc4adb75a6e0db4))

   it's chunk of bbox not frames, 48 is a good base

- 🐛 avoid inplace drawing ([91f8097](https://github.com/melMass/ComfyUI-SapiensPose/commit/91f80974930e15ae9dbe002c8a5c3b9e3d5ad786))

- 🐛 skip early conversion ([95d1f8d](https://github.com/melMass/ComfyUI-SapiensPose/commit/95d1f8df8f2968a8f8cc69260a9f7e084f9ee667))

- 🐛 do not store intermediary crop tensors ([87ae791](https://github.com/melMass/ComfyUI-SapiensPose/commit/87ae7918f38675454c269b42edfaa251f18e6438))

   this blows ram otherwise, we will do it on the fly<br>in estimate_pose

- 🐛 remove basic log config ([3ef4982](https://github.com/melMass/ComfyUI-SapiensPose/commit/3ef49823dca679789433d85688515318deddc518))

   was used in dev

- 🐛 remove dev dependencies ([3b89316](https://github.com/melMass/ComfyUI-SapiensPose/commit/3b8931672686a400c74f73994279c9c9386e6a77))

   not needed anymore, also fix a typo

- 🐛 ensure fp32 ([e2822b5](https://github.com/melMass/ComfyUI-SapiensPose/commit/e2822b5cb1d34419bd80a5ec744dff04dcfa2024))

   when a bbox isn't found we use the full frame.<br>The array wasn't in the proper dtype


### Documentation

- 📚 update readme ([2c7f2b6](https://github.com/melMass/ComfyUI-SapiensPose/commit/2c7f2b6a9a65da60247f086a698eb8eb35f2492a))

- 📚 remove todo from readme ([66e59ed](https://github.com/melMass/ComfyUI-SapiensPose/commit/66e59ed21e0fa67f3e5f7ffd2b09ec4cc9f78fd6))

   now using the issue #1 to track this

- 📚 WIP readme ([4c5380f](https://github.com/melMass/ComfyUI-SapiensPose/commit/4c5380f937753f0337bdbdefeca38447b8621bdf))


### Features

- [**breaking**] ✨ bbox_padding and bone skipping ([8ac9b35](https://github.com/melMass/ComfyUI-SapiensPose/commit/8ac9b356949226846c7043a29e1a5db693e41b61))

- ✨ add smooth node ([b4d1f92](https://github.com/melMass/ComfyUI-SapiensPose/commit/b4d1f92ee6e7c563970c79bc903d572c5497229a))

- ✨ re-expose alternative methods for extraction ([404f149](https://github.com/melMass/ComfyUI-SapiensPose/commit/404f1499e36e1779ddf9b207574ac2a5912ec881))

   -uses the same logic I tried before just refactored to better<br>compare them.<br>- expose label drawing options.

- ✨ add I/O nodes ([f30eaeb](https://github.com/melMass/ComfyUI-SapiensPose/commit/f30eaebc4d6c5a7b3b5e6a9eb88c78e46113c52c))

   to save and load keypoints.<br>It supports msgpack and json based on the extension.<br>msgpack is recommended but require extra dependencies

- ✨ unchunk the data out of pose estimator ([4ad5652](https://github.com/melMass/ComfyUI-SapiensPose/commit/4ad5652b07c162f0c8c999fa7e6475a9ad74aa53))

   This makes pose estimator output a list of total_frame_count instead<br>of total_frame_count / chunks_size.<br>This make bboxes optional for drawing now and a better layout for I/O

- ✨ add back autoscale ([47dc3ae](https://github.com/melMass/ComfyUI-SapiensPose/commit/47dc3aeacb76828bc077c229791d4c2d6c8b0e69))

- ✨ add back proxytensor from mtb ([22d44cd](https://github.com/melMass/ComfyUI-SapiensPose/commit/22d44cd96241d86e2601640ff560e150ddffc6da))

- ✨ add debug helper nodes ([6d85dbc](https://github.com/melMass/ComfyUI-SapiensPose/commit/6d85dbcdd2377a63b9f827ee399e2e3377051d64))

   - Very crude massive toggle<br>- Preview Crops

- ✨ update comfy nodes ([8a56e78](https://github.com/melMass/ComfyUI-SapiensPose/commit/8a56e782ba21a273bdf4a622079e40a28f2170ac))

- ✨ update type annotation and rework drawing ([8b83196](https://github.com/melMass/ComfyUI-SapiensPose/commit/8b83196a4362811a6166f1c3e9759d7b428713ec))

- ✨ soft remove yolo and rework the data flow ([8ef9d70](https://github.com/melMass/ComfyUI-SapiensPose/commit/8ef9d700f24b1b4c8b8420dcbe856ff2c9290482))

   yolo can easily be added back later

- ✨ update palette classes ([67133ad](https://github.com/melMass/ComfyUI-SapiensPose/commit/67133adcd10ca09869291fbdc2e40422b7b96d94))

- ✨ add draw_labels param ([2b642f5](https://github.com/melMass/ComfyUI-SapiensPose/commit/2b642f5f5e7f709c0fc0eaf9f0e56dd934a0ee44))

   to more easily debug keypoints

- ✨ add CLI ([b9c12ff](https://github.com/melMass/ComfyUI-SapiensPose/commit/b9c12fff27da64ce52fb4a4b3ea78cb833ef021b))

- ✨ add LazyProxyTensor ([4d1b6f4](https://github.com/melMass/ComfyUI-SapiensPose/commit/4d1b6f4044004ee6d5396abfa6713f0c41131e39))

   copied from mtb

- ✨ simplify draw to handle new data layout ([5bd93a2](https://github.com/melMass/ComfyUI-SapiensPose/commit/5bd93a25114ae4c1e45a2e060c02bbd4d0563c71))

- ✨ on the fly cropping from estimate_pose ([2676751](https://github.com/melMass/ComfyUI-SapiensPose/commit/2676751efd54a0d5fa5b27a52825c23bf9aced4d))

- ✨ pass base_path for cli ([c954f85](https://github.com/melMass/ComfyUI-SapiensPose/commit/c954f851926215f9821bfce0bb2b5b34d6cfcf2e))

- ✨ use chunks for bbox detection v1 ([c44eac4](https://github.com/melMass/ComfyUI-SapiensPose/commit/c44eac4b5038f78390ce3133013cdec02f8468b3))

- ✨ migrated ([72d316c](https://github.com/melMass/ComfyUI-SapiensPose/commit/72d316cd1e06e5b42cfa40105830a7eb841fedd0))


### Miscellaneous Tasks

- 🧹 add util to plot "animation curves" ([41a9e64](https://github.com/melMass/ComfyUI-SapiensPose/commit/41a9e6436034a7bb24035339fdd761b8c8923132))

   either by bone name or bone id

- 🧹 remove prints ([ac8a6e8](https://github.com/melMass/ComfyUI-SapiensPose/commit/ac8a6e8206667d20884b4782e4ec15936569d133))

- 🧹 prop drill draw_bbox ([e353fab](https://github.com/melMass/ComfyUI-SapiensPose/commit/e353fabed77f855b741cdbd991d14da2f1800138))

- 🧹 simplify types ([c583376](https://github.com/melMass/ComfyUI-SapiensPose/commit/c583376e4970d48e197cdbc2bd274698e43c6661))

- 🧹 add typing dependencies ([ba6c6a6](https://github.com/melMass/ComfyUI-SapiensPose/commit/ba6c6a66831abbb494c3d158e2c6bbff77bca227))

   I want to either remove them completely or find a way to<br>have them optionally

- 🧹 update requirements ([932d5d1](https://github.com/melMass/ComfyUI-SapiensPose/commit/932d5d11dc02b6d034aa25ec65da0c5bdd53e23a))

   for now pretty strict, so for non windows/3.11 users it will<br>  fallback to YOLO


### V1.0

- Base version ([00cbcd9](https://github.com/melMass/ComfyUI-SapiensPose/commit/00cbcd9d51ecbdfd229709721ad6ff55c3f36a51))

   this consolidates the beta version more and remove jaxtypings/beartypes.


### Wip

- 🚧 start flattening data ([01aadc3](https://github.com/melMass/ComfyUI-SapiensPose/commit/01aadc3a2d7692247f6f66d402df8492e3ffe240))

   most of the v2 pipeline was batched but it's not fully anymore.<br>the goal is to get a list[KeypointResult] with 1 keypointResult per frame.<br><br>Currently they are chunked by chunk_size

- 🚧initial ([23ad506](https://github.com/melMass/ComfyUI-SapiensPose/commit/23ad5068adf976a34bd6107b7f69f88865777a34))



