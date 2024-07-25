from rlkit.samplers.eval_suite.eval_suite import EvalTest
import numpy as np
import pandas as pd
import os
from rlkit.samplers.eval_suite.utils import create_regular_image_directories, save_regular_images
import copy
import cv2


class SuccessRateTest(EvalTest):
    def single_evaluation(self, eval_number: int) -> dict:
        print(f"{self.name} eval", eval_number)
        trajectory_log = pd.DataFrame()
        save_images = (self.epoch % self.save_images_every_epoch == 0) and (
            eval_number == 0)

        if save_images:
            create_regular_image_directories(
                self.base_save_folder, self.name, self.epoch)
            blurred_path = f"{self.base_save_folder}/epochs/{self.epoch}/{self.name}/blurred_cnn"
            os.makedirs(blurred_path, exist_ok=True)

        path_length = 0
        o = self.env.reset()
        d = False
        success = False

        #*********바꾼부분**************************#
        # while path_length < 100:
        while path_length < self.max_path_length:
            o_for_agent = self.obs_preprocessor(o)
            a, _, aux_output = self.policy.get_action(o_for_agent)

            if 'image' in o.keys() and self.save_blurred_images and save_images:
                image = o['image']
                image = image.reshape((-1, 100, 100))*255
                cv2.imwrite(
                    f"{self.base_save_folder}/epochs/{self.epoch}/{self.name}/blurred_cnn/{str(path_length).zfill(3)}.png", image[0])
                
            #**********바꾼부분**********************#
            # if save_images:
            #     save_regular_images(
            #         self.env, self.base_save_folder, self.name, self.epoch, path_length, aux_output)
            save_regular_images(
                self.env, self.base_save_folder, self.name, self.epoch, path_length, aux_output)
            print("path_length: ", path_length)

            next_o, r, d, env_info = self.env.step(copy.deepcopy(a))

            trajectory_log_entry = self.env.get_trajectory_log_entry()
            trajectory_log_entry["is_success"] = env_info['is_success']
            trajectory_log = trajectory_log.append(
                trajectory_log_entry, ignore_index=True)

            path_length += 1
            if env_info['is_success']:
                success = True

            if d:
                break
            o = next_o

        trajectory_log.to_csv(
            f"{self.base_save_folder}/epochs/{self.epoch}/{eval_number}_{self.name}_trajectory.csv")
        raw_actions = np.stack(trajectory_log['raw_action'].values)
        np.savetxt(f"{self.base_save_folder}/epochs/{self.epoch}/{eval_number}_{self.name}_executable_raw_actions.csv",
                   raw_actions, delimiter=",", fmt='%f')
        corner_distances = np.linalg.norm(
            next_o['achieved_goal']-next_o['desired_goal'])

        score_dict = {"corner_distance": corner_distances, "success_rate": 0.0,
                      "corner_0": 0.0, "corner_1": 0.0, "corner_2": 0.0, "corner_3": 0.0}
        for info_key in env_info.keys():
            if "corner" in info_key and not info_key == "corner_positions":
                score_dict[info_key] = env_info[info_key]
        if success:
            score_dict["success_rate"] = 1.0

        return score_dict
