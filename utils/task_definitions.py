

constraints = dict(
                    # origin: robot의 현재 position
                        #robot의 initial joint value인 self.initial_qpos를 이용해서 현재 x,y위치를 구하는 방식으로 하던가 해야 할 듯.
                    # target: 모든 configuration을 하지는 않고, N개의 점을 찍는 것으로 하기
                    random_25 = lambda min, mid, max, dist, origin_x,origin_y: [
                                        dict(origin=f"S{origin_x}_{origin_y}", target=f"S{x}_{y}", distance=dist) for x in range(min, max + 1, mid) for y in range(min, max + 1, mid)
                                        ],

                    diagonal = lambda min, mid, max, dist: [
                                   dict(origin=f"S{max}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(1, 1, 0)),
                                   dict(origin=f"S{min}_{min}", target=f"S{min}_{min}",distance=dist),
                                   dict(origin=f"S{mid}_{mid}", target=f"S{mid}_{mid}", distance=dist), ],


                    sideways = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{min}", target=f"S{max}_{min}", distance=dist),
                                        dict(origin=f"S{min}_{min}", target=f"S{min}_{min}", distance=dist)],

                    sideways_two_corners = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_one_corner = lambda min, mid, max, dist: [
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_two_corners_mid = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist)],
                   )
