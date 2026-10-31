# vb-1.1 scoring scope

Generated 2026-09-16T18:02:10+00:00 by `backend/scripts/vbench_scope.py` (H-2, D-40, D-43).

- **Source manifest:** `backend/data/vbench/manifest.parquet`, sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, 15767 rows (15492 in the pool).
- **Scope:** `backend/data/vbench/scope/vb-1.1.parquet`, sha256 `2041bf6787219d730658ecb64a19e49408b43a3a6e855fea61ec83cc785910dc`, 12141 rows.
- **Rule:** every image real, ≤ 5000 image fakes proportional over families (≥ 1 each), ≤ 350 video reals even over sources (D-43), ≤ 300 video fakes even over vendors, every audio row; robustness 300 real (R1–R5) + 300 fake images from `test` only (J-4), even over strata, under jpeg75, jpeg50, webp80, resize0.5, crop70, social_ig, social_x, screenshot_jpeg, noise3, upscale2x, exif_forge.

## Items

| Modality | Label | Pool | Scope |
|---|---|---|---|
| audio | fake | 1000 | 1000 |
| audio | real | 600 | 600 |
| image | fake | 7246 | 5000 |
| image | real | 4891 | 4891 |
| video | fake | 1320 | 300 |
| video | real | 435 | 350 |

## Image fake families

| Family | Pool | Scope |
|---|---|---|
| faceswap-stills | 700 | 483 |
| flux1 | 400 | 276 |
| flux2 | 256 | 177 |
| gan-faces | 400 | 276 |
| gpt-4o-image-1 | 244 | 168 |
| gpt-image-1.5-2 | 300 | 207 |
| hidream | 140 | 97 |
| hunyuanimage | 64 | 44 |
| ideogram | 154 | 106 |
| imagen3 | 150 | 103 |
| imagen4 | 258 | 178 |
| legacy | 400 | 276 |
| long-tail | 500 | 345 |
| midjourney | 402 | 277 |
| modern-t2i-faces | 295 | 204 |
| nano-banana | 400 | 276 |
| partial-edits | 600 | 414 |
| personalised | 300 | 207 |
| qwen-image | 275 | 190 |
| recraft | 130 | 90 |
| sd3 | 200 | 138 |
| seedream3 | 88 | 61 |
| seedream4 | 290 | 200 |
| wild | 300 | 207 |

## Video fake families

| Family | Pool | Scope |
|---|---|---|
| ai-talking-humans | 100 | 19 |
| animatediff | 30 | 19 |
| cogvideox | 30 | 19 |
| kling | 112 | 19 |
| luma-ray | 30 | 19 |
| marey | 28 | 19 |
| mochi | 20 | 19 |
| pika | 20 | 19 |
| runway | 60 | 19 |
| seedance | 40 | 19 |
| sora | 140 | 19 |
| stable-video | 30 | 19 |
| talking-head-animation | 290 | 18 |
| veo | 200 | 18 |
| videopoet | 30 | 18 |
| wan-hunyuan | 160 | 18 |

## Real strata

| Modality | Stratum | Scope |
|---|---|---|
| audio | A-bonafide | 600 |
| image | R1 | 849 |
| image | R2 | 1024 |
| image | R3 | 1068 |
| image | R4 | 900 |
| image | R5 | 1000 |
| image | R6 | 50 |
| video | V-general | 335 |
| video | V-talking-head | 15 |

## Robustness images

| Label | Stratum or family | Items |
|---|---|---|
| fake | faceswap-stills | 14 |
| fake | flux1 | 13 |
| fake | flux2 | 13 |
| fake | gan-faces | 13 |
| fake | gpt-4o-image-1 | 13 |
| fake | gpt-image-1.5-2 | 13 |
| fake | hidream | 13 |
| fake | hunyuanimage | 13 |
| fake | ideogram | 13 |
| fake | imagen3 | 13 |
| fake | imagen4 | 13 |
| fake | legacy | 13 |
| fake | long-tail | 13 |
| fake | midjourney | 13 |
| fake | modern-t2i-faces | 13 |
| fake | nano-banana | 13 |
| fake | partial-edits | 13 |
| fake | personalised | 13 |
| fake | qwen-image | 13 |
| fake | recraft | 13 |
| fake | sd3 | 13 |
| fake | seedream3 | 13 |
| fake | seedream4 | 13 |
| real | R1 | 60 |
| real | R2 | 60 |
| real | R3 | 60 |
| real | R4 | 60 |
| real | R5 | 60 |

## Unscored video reals (D-43)

85 pool video reals are outside the scope; they stay in the manifest, unscored.

| Source | Pool | Scope | Unscored |
|---|---|---|---|
| commons-video-speeches | 15 | 15 | 0 |
| deepaction-v1 | 70 | 70 | 0 |
| openvid-1m | 350 | 265 | 85 |

- **openvid-1m:** `004a4b189bf4a59d0c21d7af3fb66de43f6449c933bc61eeb14cb810c171fad9`, `0407e1925e95fce293924adcf5216dc2d4bde8d8d5d6fd104985644ed701c0ed`, `04778815b707edea0ef273bdc4bc1ae6011c3b03297e80deb3a791b6cd1e308b`, `049f09ed75f8758e34f34a2cca19381e311bfd034de8d4cf59fd4c08841f6aae`, `0976a9cd18108015ccc1f232770a967fda610900d956d62f6d885bffa0757630`, `0a19832265ac2d46c74f5a3bf745fc2dd39e1c3243ee40c3a8bcbc237f29b0e8`, `0c6b1668fc65c6f6ac9ce7a2b48f56c721cda564c3f9b3bd8f9841ac31e94b64`, `0e7e54aedbd74fe2ee1e0790f87d88840b5ab2011e7a254a4d7705909fd8f6fb`, `0ec2fed234af65f323df806a84d46ef26a35ff8539b339f6a91f80c19a510bb3`, `12d122121e857bdd91d2a0bb8481c2f8a23baa63447fa77d3488d6142460f1df`, `16d422228c644c1b39b4ae42a68ff494dcadf2ced9f265a882b1f4d8c3f48250`, `1919706ef4e08da34efafb5443c8143b9cd958f35bb23417149ea82e34f9980d`, `2236b15b2fc61fcd35867b9e17f1e9b656bbb0790e8f4a77aaf014a186ce4942`, `2250e02cee572ad570516ef83e3a9d08c1bc8783e73fd0771ad0c902c27cb7ee`, `23064b9c9739c9392273bac8957c132fdaa1806aaa47eb53be71a0e7be319ce1`, `2544d21c8c492dec27f39bfb7b56bd6b5ab2e96065a025c344e01d7ae65eeed5`, `26f4575b5c166bb77d89709a916e58c2a906272c5ea0cd317ccc7349dff7706d`, `2fc4541229f537a8ec3ad6171c288779d9732a035fdc5b99213417c4fc1cebff`, `353ee2a6a08ddaea519f64cb64c65c85ac6c170c6be2eb5cec2ff298277b3fd4`, `356ddbf5940684bcb670469d8f35964b08669b4922d936d71976c8402b0f71b4`, `35d11c0591ef5bb7c0226beb965991565161d05e06bc1b2869dae93d60d2b147`, `37d82540ba5c2892e524086f3f37b67fa92012bd0c1d59b07a017347e9ff7302`, `38809ef1330422a6a3523d99f165a40c59ce7cd13e44375d36ddc7dee71be351`, `3a1b1cfb5a3dfdbdc71a4d8001491a6876f6c547656a827d741877c347ee116c`, `406c40aeab56778cbe1a7d178ce8abe55255e38c03899f88885cfe6ba588285a`, `4563e9a36b75198d9a5c1b524dcc054cd8731cca26536f8850e9a3f782fe7f17`, `45d5d2dc21acbf513be48bc872447b6f511811735092ed6afb173c49883485bd`, `49d6ef9f48beafaf53a7031b353a4fdef2f72026ce6e9e7e0dc6090827af3fd9`, `4faf1972e394fb84b1747fc98994ea0ba447f06eb638fadbe37a413eea755624`, `50eb720c064bcd179c1225781cccee6874ed4801a2773c6673267a259dc25932`, `52f130a5be590e8b431dd1d77403441a2ec3f3b6a3d5d3a975226c30827d5cb4`, `561938bb9fa731fc6f5090706c6a74020e1d088fceffa5e24fd539fafa69ed67`, `583e40249679b0eba81d6290c83986b3b7b96b1025de957c382b2a2fda9d56fc`, `5c4f2839fe07a1224b595526238ef2f3042361c55a8164f88bfc05ae5ca1aac8`, `606f4732d322142cd22afbae140dad4fb9e5195bddb84da6af26eb495cc90a51`, `657ea07b5ff4b4d1b84693fe2c5851cac99e41f7d61236b2a672667941f0c5ab`, `65c87526029f3e94da756d442c3d195ab561a919a5567b8358d98a226b25a835`, `67d116501a8da7265b3e6e7a86ec3940ca750903229e535d480386a72910a871`, `68d05e5a178064e420b3c907cd34da1152822bd078e048db4d249fe1f5660851`, `691bc73335f4794f9cf9adf5a857bc25e597bfa35aa821b3ac2ca0b42ea14c15`, `6e375a10b6e25f55b2583e92eb518f2d258b4d7a06f4443b3f7c3984e2a5acba`, `6f7a8f74367d81b6cd26b8180fb0a3a2195530f478fcf30475789b5a8b4a4b60`, `71836132f53d3ef169c8c93983599e6073aa9482cb1a7dafd63d490008e17528`, `77e6a235705946f37c31490d6b3160ac813e3fa0d3e7783f93c21e3c25aa1719`, `7ace740075f95df863c1f9576e54d3d8c229bffaf9064823d56459c8f3c9d6f5`, `7bd1e3a74d97223adb1dd1aa725e3ed1037706f2264609a852369c4432cc2399`, `7c5065e8db702e281230703d8ee545f2c6ac46ebbcc43e81b76e389209d03702`, `7f7d6a5cd4636f80cf9445ff10421ebba38661a44f4a5d88016a1a1cf5364b18`, `852256d9f33be5b2dabea8dfdc5cf8b5226d0b8f54fb9df606e0381671b739ce`, `88b06cfd59d485950b675f4e3cb21b0879a556fb13a5eebee00e062781a76944`, `89f6f313694408bcbff82ecc8aee2b540d744650dcf53c40ca89cad414d8bfe1`, `8cbc361788c01fd7c4e9f70bdba0283fdeb953842668dc7e80574a7fec4b7421`, `8dff5fd494e183c2c0ceec9b82b291ac9566a92b6d875330522a7b5d387bfabc`, `8f76206e1bd8ab90d4667d7335a9cfb08e778d506c5031316b91ba693ff743ba`, `94b7a50ea2f66366e74258d94cc7341cc142d9b92f235212f49b460f8ce7e95e`, `a0e2dd7e1969f23823a76d79031eaa13b9e6fa2bbc2da3a9aa83a90fd9148eb1`, `a19eb67dcff144f248a321c805821aa54ba33102d8d1555415e5fc805bdb0ff9`, `ab06dd992b24e7fcd362e27fb9cdce310777c7c8acf1539fff4a8cb5a3ef63d0`, `ace8e1014b95ecee715428de7450707ae19e025c0431d28f833e6b8226f5eacb`, `af89d06de3ef2a0aab0c561125c83bec90f83f43da871db8e4d8648e242a88a4`, `bb7a239fd8e6d7e45d39f5918baba569d3c4f88e2b71089f15ab7b77ed0e0306`, `c532e5969aa766aa0b75c6674c5dff270b3ce3c7c70814e2b2a0842328881133`, `ca4aa66ae4307189c98b9ad7ff8088d002cb127e8e70c6e9abec42d8f08f39f8`, `cec777a36851357fe9d8245f53a8b8b20e2642eea0ef5dd416090d0bdb4aee42`, `cfb23a64359ccf1811565878946c092378843e6f24daf7527e452e86d1cf4730`, `d04ae5312aa57ff8f88b9dbee75467a742ade610ff7902f4a64196d9bf6650d8`, `d1bbe37f68c9f62b2b8ce45673500635855b5224035664779f9a0947b70fa918`, `d234770f867ef797a4ca5476b4eaafefcf4313b794d215be46e638db00ff972b`, `d541a7dd62837c504d0b9fed9d84c52c7a9e2c1b4b572b459557588a22bb798f`, `d7ae08dacbd29793c5b183cb7f24807d2402f9c34b4e930ad726cd378f6aef1d`, `d9539573152280babe98a96a5f64d6f094e0320518ffae1c555c7b4dee89361f`, `dd72b2705bf92732929c05b29c7d00b15c0397848d70af34334379591c14a6f0`, `de49372a5febf7be02c08968f2965300f3e9309da8dd410ad36f5b267c2a3015`, `dea013a2d8e84726882117608da43c4f5119bd028f193cf455555336dcdef9a0`, `e0cac9e9fb2d2d78940ce9db14b9fa261165ed843e08eaa198423fd6f1b29f3f`, `e24eb86616823f0e9437a22c5d40fdfc966aa9c040b0a0d6bac5a5ee56b36447`, `e7157b6ea8c2cc725ba5805ba4bb31a0ae24114ff61526e2c75aad8b0f88e938`, `e83cc7cf6086246cd940be84850b08fa2c996170f32cec68881b5b85ae72d904`, `ecfa77e4a7586e1905534cbf4f6ca8a856e63beb0486e5d0b56c1542ebd563ee`, `ef8fcf521dd2cdd53677f6e2d0c8d41c5fcb6fa11d388fc34bdf211657284a9d`, `f094d7dd77275d6dd4e7690d8e32d4e3192e450d14670dd9380aa5cbd2aeb33f`, `f8d05f23779346572407477e6b01dcf3648a0a93e329608619952e1d5443b4c2`, `fbb5eb60f955059cba593abd9f761af28dc89c608028a46c3099cb179564aefd`, `fdd57ecf5e28c27dfe81cc895d9ca87e8810b71548f9807ac5504f714b0b43a9`, `fe199c1c774bf7e2b535432b72427533e7c5125e7f41eac1bbf567c16bef7617`
