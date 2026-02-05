from .schema import CourseKnowledgeJSON, Topic, Subtopic


def normalize_course(
    course_title: str, subtopics: list, source_files: list
) -> CourseKnowledgeJSON:
    """
    Converts subtopics into a normalized JSON object.
    """
    # Convert dict subtopics to Subtopic objects
    subtopic_objects = []
    for sub in subtopics:
        subtopic_objects.append(Subtopic(**sub))

    topic = Topic(
        id="topic_1",
        title=course_title,
        summary=" ".join(sub.summary for sub in subtopic_objects if sub.summary)[
            :1000
        ],  # Combine summaries
        subtopics=subtopic_objects,
    )

    course_json = CourseKnowledgeJSON(
        course_title=course_title, source_files=source_files, topics=[topic]
    )

    return course_json
