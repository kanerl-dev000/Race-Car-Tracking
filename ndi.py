import ndi


def main():
    # Initialize the NDI runtime
    ndi.initialize()

    # Create a finder to discover NDI sources
    finder = ndi.finder_create_v2()

    # Wait for sources to be discovered
    ndi.finder_wait_for_sources(finder, 1000)

    # Get the list of discovered NDI sources
    sources = ndi.finder_get_current_sources(finder)

    if len(sources) == 0:
        print("No NDI sources found.")
        return

    # Choose an NDI source (you can adjust the index)
    chosen_source = sources[0]

    # Create an NDI receiver
    receiver = ndi.Recv()

    # Connect to the chosen NDI source
    receiver.create(chosen_source)

    try:
        while True:
            # Receive a video frame
            frame = receiver.capture()

            if frame is not None:
                # Process or display the received frame
                pass
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        receiver.destroy()
        ndi.destroy()


if __name__ == "__main__":
    main()
