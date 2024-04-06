import os
import pandas as pd


class Eval_Formatter:

    def __init__(self, execution_output_folder):
        self.data_processing_columns = [
            "Patient_ID",
            "Pop_Epoch_Size",
            "Pop_Fold",
            "Ind_Epoch_Size",
            "Ind_Fold",
        ]

        project_root_folder = "Research-Project"
        root_dir = self.update_cwd(project_root_folder)

        main_output_folder = "Output_Results"

        self.RESULT_DIR = f"{root_dir}/{main_output_folder}/{execution_output_folder}"

        self.POPULATION_EPOCH_SIZE_LIST = [100, 300, 500, 700, 1000, 2000, 3000]
        self.PATIENT_EPOCH_SIZE_LIST = [100, 300, 500, 700, 1000, 2000, 3000]

        # Used for individual model
        self.ind_target_grp = [
            "SB-001",
            "SB-003",
            "SB-008",
            "SB-011",
            "SB-012",
            "SB-022",
            "SB-023",
            "SB-025",
            "SB-028",
            "SB-037",
            "SB-043",
            "SB-056",
            "SB-059",
            "SB-060",
            "SB-070",
            "SB-071",
            "SB-073",
            "SB-078",
            "SB-079",
            "SB-080",
            "SB-082",
            "SB-083",
            "SB-112",
        ]

    def update_cwd(self, project_root_folder):
        cwd = os.getcwd()

        # Loop until we find the "research-project" directory or reach the root
        while os.path.basename(cwd) != project_root_folder and cwd != os.path.dirname(
            cwd
        ):
            cwd = os.path.dirname(cwd)

            # Check if "Research-Project" directory is found
        if os.path.basename(cwd) == project_root_folder:
            os.chdir(cwd)
            return os.getcwd()

    def segregate_identifier(
        self,
        index: str,
        column_return_type: str,
        num_pop_folds: int,
        num_ind_folds: int,
    ) -> int:
        index = str(index).upper()
        if column_return_type == "Pop_Epoch_Size":
            for i in self.POPULATION_EPOCH_SIZE_LIST:
                if str(i) + "_POP" in index:
                    return i
        elif column_return_type == "Ind_Epoch_Size":
            for j in self.PATIENT_EPOCH_SIZE_LIST:
                if str(j) + "_IND" in index:
                    return j
        elif column_return_type == "Patient_ID":
            for i in self.ind_target_grp:
                if str(i) in index:
                    return i
        elif column_return_type == "Pop_Fold":
            for i in [*range(1, num_pop_folds + 1)]:
                if "POP_FOLD_" + str(i) in index:
                    return i
        elif column_return_type == "Ind_Fold":
            for i in [*range(1, num_ind_folds + 1)]:
                if "IND_FOLD_" + str(i) in index:
                    return i

    def load_csv_to_df(
        self,
        file_path: str,
        num_pop_folds: int,
        num_ind_folds: int,
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path, index_col=0)
        for i in self.data_processing_columns:
            df[i] = df.index.map(
                lambda x: self.segregate_identifier(
                    x,
                    i,
                    num_pop_folds,
                    num_ind_folds,
                )
            )

        return df

    def resave_eval_df(
        self,
        eval_data_type: str,
        Architecture_GROUP: str,
        bool_remove_target: bool,
        num_pop_folds: int,
        num_ind_folds: int,
    ) -> None:
        if bool_remove_target:
            suffix = "_evaluation_metrics_TargetRemoved.csv"
        else:
            suffix = "_evaluation_metrics.csv"

        if eval_data_type == "individual":
            eval_data_path = "/diagrams_n_individual_models/Ind_"
        elif eval_data_type == "population":
            eval_data_path = "/diagrams_n_population_models/Pop_"

        file_path = f"{self.RESULT_DIR}/{Architecture_GROUP}{eval_data_path}{Architecture_GROUP}{suffix}"

        if os.path.isfile(file_path):
            temp_unsorted_eval_data_df = self.load_csv_to_df(
                file_path,
                num_pop_folds,
                num_ind_folds,
            )

            temp_sorted_eval_data_df = temp_unsorted_eval_data_df.sort_values(
                by=self.data_processing_columns, ascending=True
            ).drop(columns=self.data_processing_columns, errors="ignore")

            output_path = os.path.join(file_path)
            temp_sorted_eval_data_df.to_csv(output_path, mode="w")


def main():

    # SETTING GLOBALS AND FILEPATHS

    # Possible Values:
    Architecture_GROUP_LIST = [
        "TL_1_Frozen",
        "TL_2_Frozen",
        "TL_3_Frozen",
        "TL_All_Frozen",
        "TL_All_Unfrozen",
    ]

    execution_output_folder = "SDDSmoteBoost_Unf_All_Patient"

    # Number of Folds for running Pop Model
    # Possible values : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_pop_folds = 5

    # Number of Folds for running Ind Model
    # Possible values : [3, 4, 5]
    num_ind_folds = 3

    eval_formatter = Eval_Formatter(execution_output_folder)

    bool_remove_target_list = [True, False]
    eval_data_type_list = ["individual", "population"]
    for eval_data_type in eval_data_type_list:
        for bool_remove_target in bool_remove_target_list:
            for Architecture_GROUP in Architecture_GROUP_LIST:
                eval_formatter.resave_eval_df(
                    eval_data_type,
                    Architecture_GROUP,
                    bool_remove_target,
                    num_pop_folds,
                    num_ind_folds,
                )


if __name__ == "__main__":
    main()
