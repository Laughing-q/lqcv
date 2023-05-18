from lqcv.tools import remove_extra_files, videos2images, similarity

if __name__ == "__main__":
    remove_extra_files("runs/more/", "runs/less/", "runs/target", reverse=False)
